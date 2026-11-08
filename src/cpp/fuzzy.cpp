#include "fuzzy.hpp"

#include <omp.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace aof {

namespace {
constexpr double kSmoothKTolerance = 1e-5;
constexpr double kMinKDistScale = 1e-3;
}  // namespace

FuzzyResult fuzzy_simplicial_set(const int64_t* knn_indices, const float* knn_dists, int64_t n,
                                 int64_t k, double n_neighbors, double local_connectivity,
                                 double set_op_mix_ratio, double bandwidth) {
    FuzzyResult r;
    r.sigmas.assign(n, 0.f);
    r.rhos.assign(n, 0.f);
    const double target = std::log2(n_neighbors) * bandwidth;

    // np.mean(distances) in numba: float32 accumulation in row-major order.
    float total = 0.f;
    for (int64_t e = 0; e < n * k; ++e) total += knn_dists[e];
    const double mean_distances = (double)total / (double)(n * k);

    // ---- smooth_knn_dist ----------------------------------------------------
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        const float* row = knn_dists + i * k;
        float rho = 0.f;
        // rho from the non-zero distances (row order).
        int64_t nnz = 0;
        for (int64_t j = 0; j < k; ++j) nnz += row[j] > 0.f;
        if ((double)nnz >= local_connectivity) {
            const int64_t index = (int64_t)std::floor(local_connectivity);
            const double interpolation = local_connectivity - (double)index;
            auto nz = [&](int64_t want) {  // want-th non-zero distance
                int64_t c = 0;
                for (int64_t j = 0; j < k; ++j)
                    if (row[j] > 0.f) {
                        if (c == want) return row[j];
                        ++c;
                    }
                return 0.f;
            };
            if (index > 0) {
                rho = nz(index - 1);
                if (interpolation > kSmoothKTolerance)
                    rho = (float)((double)rho + interpolation * (double)(nz(index) - nz(index - 1)));
            } else {
                rho = (float)(interpolation * (double)nz(0));
            }
        } else if (nnz > 0) {
            for (int64_t j = 0; j < k; ++j) rho = std::max(rho, row[j]);
        }
        // Binary search.  umap declares these locals float32, but numba 0.5x/0.6x
        // types them as float64 (verified in the generated LLVM IR): only
        // d = dist - rho is a float32 subtraction; d / mid, exp and the sum are
        // double, and the result is stored as float32.
        double lo = 0.0, hi = (double)FLT_MAX, mid = 1.0;
        for (int it = 0; it < 64; ++it) {
            double psum = 0.0;
            for (int64_t j = 1; j < k; ++j) {
                const float d = row[j] - rho;
                if (d > 0.f)
                    psum += std::exp(-(double)d / mid);
                else
                    psum += 1.0;
            }
            if (std::fabs(psum - target) < kSmoothKTolerance) break;
            if (psum > target) {
                hi = mid;
                mid = (lo + hi) / 2.0;
            } else {
                lo = mid;
                if (hi >= (double)FLT_MAX)
                    mid *= 2;
                else
                    mid = (lo + hi) / 2.0;
            }
        }
        float sigma = (float)mid;
        if (rho > 0.f) {
            float s = 0.f;
            for (int64_t j = 0; j < k; ++j) s += row[j];
            const double mean_ith = (double)s / (double)k;
            if ((double)sigma < kMinKDistScale * mean_ith) sigma = (float)(kMinKDistScale * mean_ith);
        } else {
            if ((double)sigma < kMinKDistScale * mean_distances)
                sigma = (float)(kMinKDistScale * mean_distances);
        }
        r.sigmas[i] = sigma;
        r.rhos[i] = rho;
    }

    // ---- membership strengths, one row per cell (sorted by column) ------------
    std::vector<int64_t> wptr(n + 1, 0);
    std::vector<int32_t> wcol((size_t)n * k);
    std::vector<float> wval((size_t)n * k);
    std::vector<int64_t> cnt(n, 0);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        std::vector<std::pair<int32_t, float>> e;
        e.reserve(k);
        const float rho = r.rhos[i], sigma = r.sigmas[i];
        for (int64_t j = 0; j < k; ++j) {
            const int64_t c = knn_indices[i * k + j];
            if (c < 0) continue;
            float val;
            if (c == i)
                val = 0.f;
            else if (knn_dists[i * k + j] - rho <= 0.f || sigma == 0.f)
                val = 1.f;
            else
                val = std::exp(-((knn_dists[i * k + j] - rho) / sigma));
            if (val != 0.f) e.push_back({(int32_t)c, val});  // coo -> eliminate_zeros
        }
        std::sort(e.begin(), e.end(), [](auto& a, auto& b) { return a.first < b.first; });
        // duplicate columns would be summed by scipy's coo -> csr
        int64_t m = 0;
        for (size_t t = 0; t < e.size(); ++t) {
            if (m > 0 && wcol[i * k + m - 1] == e[t].first) {
                wval[i * k + m - 1] += e[t].second;
            } else {
                wcol[i * k + m] = e[t].first;
                wval[i * k + m] = e[t].second;
                ++m;
            }
        }
        cnt[i] = m;
    }
    // ---- transpose (stable counting sort by column) ----------------------------
    std::vector<int64_t> tptr(n + 1, 0);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t t = 0; t < cnt[i]; ++t) ++tptr[wcol[i * k + t] + 1];
    for (int64_t c = 0; c < n; ++c) tptr[c + 1] += tptr[c];
    std::vector<int32_t> tcol(tptr[n]);
    std::vector<float> tval(tptr[n]);
    {
        std::vector<int64_t> pos(tptr.begin(), tptr.end() - 1);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t t = 0; t < cnt[i]; ++t) {
                const int64_t c = wcol[i * k + t];
                tcol[pos[c]] = (int32_t)i;
                tval[pos[c]] = wval[i * k + t];
                ++pos[c];
            }
    }
    // ---- fuzzy union, row by row (two sorted lists) ----------------------------
    const float mix = (float)set_op_mix_ratio, inv = (float)(1.0 - set_op_mix_ratio);
    std::vector<int64_t> outcnt(n, 0);
    auto merge_row = [&](int64_t i, int32_t* oc, float* ov) {
        int64_t a = 0, b = tptr[i], ae = cnt[i], be = tptr[i + 1], m = 0;
        const int32_t* ac = &wcol[i * k];
        const float* av = &wval[i * k];
        while (a < ae || b < be) {
            int32_t col;
            float x = 0.f, y = 0.f;
            bool both = false;
            if (b >= be || (a < ae && ac[a] < tcol[b])) {
                col = ac[a]; x = av[a++];
            } else if (a >= ae || tcol[b] < ac[a]) {
                col = tcol[b]; y = tval[b++];
            } else {
                col = ac[a]; x = av[a++]; y = tval[b++]; both = true;
            }
            // scipy: mix * ((A + At) - A.multiply(At)) + (1 - mix) * A.multiply(At)
            const float s = both ? x + y : (x + y);
            const float p = both ? x * y : 0.f;
            float v = both ? s - p : s;
            v = mix * v;
            if (both) v = v + inv * p;
            if (v != 0.f) {
                if (oc) { oc[m] = col; ov[m] = v; }
                ++m;
            }
        }
        return m;
    };
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) outcnt[i] = merge_row(i, nullptr, nullptr);
    r.indptr.assign(n + 1, 0);
    for (int64_t i = 0; i < n; ++i) r.indptr[i + 1] = r.indptr[i] + outcnt[i];
    r.indices.resize(r.indptr[n]);
    r.data.resize(r.indptr[n]);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i)
        merge_row(i, r.indices.data() + r.indptr[i], r.data.data() + r.indptr[i]);
    return r;
}

}  // namespace aof
