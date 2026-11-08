#include "knn.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace aof {

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

// out[i, j] = cn[j] - 2 <Q[i], C[:, j]> for a block of q queries and m
// candidates whose coordinates are stored transposed: Ct[t * ldc + j].
//
// Every output element is the same sequence of fused multiply-adds over t
// (acc = fma(q_t, c_t, acc)), whatever the register tiling, so the AVX-512,
// AVX2 and portable kernels below produce bit-identical results.

using DistFn = void (*)(const float* const*, int, int64_t, const float*, int64_t, int64_t,
                        const float*, float* const*);

// Portable kernel: any number of queries (<= 8) and candidates (<= 32).
static void tile_scalar(const float* const* qr, int nq, int64_t d, const float* Ct, int64_t ldc,
                        int64_t w, const float* cn, float* const* orow) {
    float acc[8][32];
    for (int i = 0; i < nq; ++i)
        for (int64_t l = 0; l < w; ++l) acc[i][l] = 0.f;
    for (int64_t t = 0; t < d; ++t) {
        const float* c = Ct + t * ldc;
        for (int i = 0; i < nq; ++i) {
            const float a = qr[i][t];
            for (int64_t l = 0; l < w; ++l) acc[i][l] = std::fma(a, c[l], acc[i][l]);
        }
    }
    for (int i = 0; i < nq; ++i)
        for (int64_t l = 0; l < w; ++l) orow[i][l] = cn[l] - 2.f * acc[i][l];
}

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
// Missing query rows are padded with row 0 (computed, never stored); missing
// candidate lanes are masked.  Lane arithmetic is unchanged by either.
__attribute__((target("avx512f")))
static void tile_avx512(const float* const* qr_in, int nq, int64_t d, const float* Ct, int64_t ldc,
                        int64_t w, const float* cn, float* const* orow) {
    const float* qr[8];
    for (int i = 0; i < 8; ++i) qr[i] = qr_in[i < nq ? i : 0];
    const __mmask16 k0 = (__mmask16)(w >= 16 ? 0xFFFF : ((1u << w) - 1));
    const __mmask16 k1 = (__mmask16)(w >= 32 ? 0xFFFF : (w > 16 ? ((1u << (w - 16)) - 1) : 0));
    __m512 a0[8], a1[8];
    for (int i = 0; i < 8; ++i) a0[i] = a1[i] = _mm512_setzero_ps();
    for (int64_t t = 0; t < d; ++t) {
        const float* c = Ct + t * ldc;
        const __m512 c0 = _mm512_maskz_loadu_ps(k0, c), c1 = _mm512_maskz_loadu_ps(k1, c + 16);
#pragma GCC unroll 8
        for (int i = 0; i < 8; ++i) {
            const __m512 q = _mm512_set1_ps(qr[i][t]);
            a0[i] = _mm512_fmadd_ps(q, c0, a0[i]);
            a1[i] = _mm512_fmadd_ps(q, c1, a1[i]);
        }
    }
    const __m512 n0 = _mm512_maskz_loadu_ps(k0, cn), n1 = _mm512_maskz_loadu_ps(k1, cn + 16),
                 m2 = _mm512_set1_ps(-2.f);
    for (int i = 0; i < nq; ++i) {
        _mm512_mask_storeu_ps(orow[i], k0, _mm512_add_ps(n0, _mm512_mul_ps(m2, a0[i])));
        _mm512_mask_storeu_ps(orow[i] + 16, k1, _mm512_add_ps(n1, _mm512_mul_ps(m2, a1[i])));
    }
}

__attribute__((target("avx2,fma")))
static void tile_avx2(const float* const* qr_in, int nq, int64_t d, const float* Ct, int64_t ldc,
                      int64_t w, const float* cn, float* const* orow) {
    const float* qr[8];
    for (int i = 0; i < 8; ++i) qr[i] = qr_in[i < nq ? i : 0];
    for (int64_t h = 0; h < w; h += 16) {
        const int64_t wh = std::min<int64_t>(16, w - h);
        alignas(32) int m0[8], m1[8];
        for (int l = 0; l < 8; ++l) { m0[l] = l < wh ? -1 : 0; m1[l] = l + 8 < wh ? -1 : 0; }
        const __m256i k0 = _mm256_load_si256((const __m256i*)m0), k1 = _mm256_load_si256((const __m256i*)m1);
        __m256 a0[8], a1[8];
        for (int i = 0; i < 8; ++i) a0[i] = a1[i] = _mm256_setzero_ps();
        for (int64_t t = 0; t < d; ++t) {
            const float* c = Ct + t * ldc + h;
            const __m256 c0 = _mm256_maskload_ps(c, k0), c1 = _mm256_maskload_ps(c + 8, k1);
#pragma GCC unroll 8
            for (int i = 0; i < 8; ++i) {
                const __m256 q = _mm256_set1_ps(qr[i][t]);
                a0[i] = _mm256_fmadd_ps(q, c0, a0[i]);
                a1[i] = _mm256_fmadd_ps(q, c1, a1[i]);
            }
        }
        const __m256 n0 = _mm256_maskload_ps(cn + h, k0), n1 = _mm256_maskload_ps(cn + h + 8, k1),
                     m2 = _mm256_set1_ps(-2.f);
        for (int i = 0; i < nq; ++i) {
            _mm256_maskstore_ps(orow[i] + h, k0, _mm256_add_ps(n0, _mm256_mul_ps(m2, a0[i])));
            _mm256_maskstore_ps(orow[i] + h + 8, k1, _mm256_add_ps(n1, _mm256_mul_ps(m2, a1[i])));
        }
    }
}
#endif

static DistFn pick_tile() {
#if defined(__x86_64__) && defined(__GNUC__)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f")) return tile_avx512;
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) return tile_avx2;
#endif
    return tile_scalar;
}

void block_dist(const float* Q, int64_t q, int64_t d, const float* Ct, int64_t ldc, int64_t m,
                const float* cn, float* out) {
    static const DistFn tile = pick_tile();
    const float* qr[8];
    float* orow[8];
    for (int64_t i = 0; i < q; i += 8) {
        const int nq = (int)std::min<int64_t>(8, q - i);
        for (int r = 0; r < nq; ++r) qr[r] = Q + (i + r) * d;
        for (int64_t j0 = 0; j0 < m; j0 += 32) {
            const int64_t w = std::min<int64_t>(32, m - j0);
            for (int r = 0; r < nq; ++r) orow[r] = out + (i + r) * m + j0;
            tile(qr, nq, d, Ct + j0, ldc, w, cn + j0, orow);
        }
    }
}

// Bounded max-heap of (distance, id); the root is the current worst.
struct Heap {
    float* d;
    int64_t* id;
    int64_t k;
    void replace_top(float dist, int64_t idx) {
        int64_t i = 0;
        for (;;) {
            int64_t l = 2 * i + 1, r = l + 1, big = i;
            float bd = dist;
            if (l < k && d[l] > bd) { big = l; bd = d[l]; }
            if (r < k && d[r] > bd) { big = r; }
            if (big == i) break;
            d[i] = d[big];
            id[i] = id[big];
            i = big;
        }
        d[i] = dist;
        id[i] = idx;
    }
};

void transpose_rows(const float* X, int64_t n, int64_t d, const int64_t* order, float* XT) {
    // XT[t * n + s] = X[order[s] * d + t]
    const int64_t B = 1024;
#pragma omp parallel for schedule(static)
    for (int64_t s0 = 0; s0 < n; s0 += B) {
        const int64_t s1 = std::min(n, s0 + B);
        for (int64_t t = 0; t < d; ++t) {
            float* dst = XT + t * n;
            for (int64_t s = s0; s < s1; ++s) dst[s] = X[order[s] * d + t];
        }
    }
}

// Nearest centroid of every row of X (ties -> lowest centroid id).
void assign(const float* X, int64_t n, int64_t d, const std::vector<float>& C, int64_t nc,
            std::vector<int32_t>& labels) {
    std::vector<float> CT((size_t)d * nc), cn(nc);
    for (int64_t c = 0; c < nc; ++c) {
        double s = 0;
        for (int64_t t = 0; t < d; ++t) {
            CT[t * nc + c] = C[c * d + t];
            s += (double)C[c * d + t] * C[c * d + t];
        }
        cn[c] = (float)s;
    }
    const int64_t QB = 256, CB = 512;
    labels.resize(n);
#pragma omp parallel
    {
        std::vector<float> out((size_t)QB * CB), best(QB);
        std::vector<int32_t> arg(QB);
#pragma omp for schedule(dynamic, 1)
        for (int64_t q0 = 0; q0 < n; q0 += QB) {
            const int64_t q = std::min(QB, n - q0);
            std::fill(best.begin(), best.begin() + q, kInf);
            for (int64_t c0 = 0; c0 < nc; c0 += CB) {
                const int64_t m = std::min(CB, nc - c0);
                block_dist(X + q0 * d, q, d, CT.data() + c0, nc, m, cn.data() + c0, out.data());
                for (int64_t i = 0; i < q; ++i) {
                    const float* o = out.data() + i * m;
                    for (int64_t j = 0; j < m; ++j)
                        if (o[j] < best[i]) { best[i] = o[j]; arg[i] = (int32_t)(c0 + j); }
                }
            }
            for (int64_t i = 0; i < q; ++i) labels[q0 + i] = arg[i];
        }
    }
}

// Rows grouped by label (counting sort, stable): offsets[c]..offsets[c+1].
void group_by_label(const std::vector<int32_t>& labels, int64_t nc, std::vector<int64_t>& offsets,
                    std::vector<int64_t>& order) {
    const int64_t n = (int64_t)labels.size();
    offsets.assign(nc + 1, 0);
    for (int32_t l : labels) ++offsets[l + 1];
    for (int64_t c = 0; c < nc; ++c) offsets[c + 1] += offsets[c];
    std::vector<int64_t> pos(offsets.begin(), offsets.end() - 1);
    order.resize(n);
    for (int64_t i = 0; i < n; ++i) order[pos[labels[i]]++] = i;
}

}  // namespace

KnnResult knn_ivf(const float* X, int64_t n, int64_t d, const KnnParams& p,
                  const std::vector<int64_t>& init) {
    const bool dbg = std::getenv("AOF_DEBUG") != nullptr;
    auto t0 = std::chrono::steady_clock::now();
    auto lap = [&](const char* what) {
        if (!dbg) return;
        auto t1 = std::chrono::steady_clock::now();
        std::fprintf(stderr, "[knn] %-10s %.3fs\n", what, std::chrono::duration<double>(t1 - t0).count());
        t0 = t1;
    };
    if (n < 2) throw std::invalid_argument("need at least 2 cells");
    const int64_t k = std::min<int64_t>(p.k, n - 1);
    int64_t nc = (int64_t)init.size();
    if (nc < 1 || nc > n) throw std::invalid_argument("bad number of initial centroids");

    // ---- k-means (Lloyd) ----------------------------------------------------
    std::vector<float> C((size_t)nc * d);
    for (int64_t c = 0; c < nc; ++c) std::memcpy(&C[c * d], X + init[c] * d, d * sizeof(float));
    std::vector<int32_t> labels;
    std::vector<int64_t> offsets, order;
    for (int it = 0; it < p.kmeans_iters; ++it) {
        assign(X, n, d, C, nc, labels);
        group_by_label(labels, nc, offsets, order);
        // Centroid = mean of members, summed in row order (deterministic).
#pragma omp parallel for schedule(dynamic, 16)
        for (int64_t c = 0; c < nc; ++c) {
            const int64_t a = offsets[c], b = offsets[c + 1];
            if (a == b) continue;  // empty: keep the previous centroid
            std::vector<double> s(d, 0.0);
            for (int64_t r = a; r < b; ++r) {
                const float* x = X + order[r] * d;
                for (int64_t t = 0; t < d; ++t) s[t] += x[t];
            }
            for (int64_t t = 0; t < d; ++t) C[c * d + t] = (float)(s[t] / (double)(b - a));
        }
    }
    assign(X, n, d, C, nc, labels);
    lap("kmeans");

    // ---- drop empty clusters, renumber along the first coordinate -------------
    {
        std::vector<int64_t> size(nc, 0);
        for (int32_t l : labels) ++size[l];
        std::vector<int64_t> keep;
        for (int64_t c = 0; c < nc; ++c)
            if (size[c] > 0) keep.push_back(c);
        std::stable_sort(keep.begin(), keep.end(),
                         [&](int64_t a, int64_t b) { return C[a * d] < C[b * d]; });
        std::vector<int32_t> remap(nc, -1);
        std::vector<float> C2(keep.size() * d);
        for (size_t i = 0; i < keep.size(); ++i) {
            remap[keep[i]] = (int32_t)i;
            std::memcpy(&C2[i * d], &C[keep[i] * d], d * sizeof(float));
        }
        for (auto& l : labels) l = remap[l];
        C.swap(C2);
        nc = (int64_t)keep.size();
    }
    group_by_label(labels, nc, offsets, order);  // order[s] = original row at sorted position s
    std::vector<int64_t> where(n);               // sorted position of each original row
    for (int64_t s = 0; s < n; ++s) where[order[s]] = s;

    // Probe enough clusters that even the smallest ones hold k + 1 cells.
    int64_t P = std::min<int64_t>(std::max<int64_t>(p.n_probe, 1), nc);
    {
        std::vector<int64_t> sz(nc);
        for (int64_t c = 0; c < nc; ++c) sz[c] = offsets[c + 1] - offsets[c];
        std::sort(sz.begin(), sz.end());
        int64_t acc = 0, need = 0;
        while (need < nc && acc < k + 1) acc += sz[need++];
        P = std::max(P, need);
    }

    // ---- data in sorted order ---------------------------------------------------
    std::vector<float> XT((size_t)d * n), xn(n);
    transpose_rows(X, n, d, order.data(), XT.data());
#pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < n; ++s) {
        const float* x = X + order[s] * d;
        double v = 0;
        for (int64_t t = 0; t < d; ++t) v += (double)x[t] * x[t];
        xn[s] = (float)v;
    }
    std::vector<float> CT((size_t)d * nc), cn(nc);
    for (int64_t c = 0; c < nc; ++c) {
        double v = 0;
        for (int64_t t = 0; t < d; ++t) {
            CT[t * nc + c] = C[c * d + t];
            v += (double)C[c * d + t] * C[c * d + t];
        }
        cn[c] = (float)v;
    }

    lap("layout");
    KnnResult res;
    res.n_clusters = nc;
    res.n_probe = P;
    res.indices.assign((size_t)n * k, -1);
    res.distances.assign((size_t)n * k, kInf);
    const int64_t B = std::max<int64_t>(1, p.block);
    const int64_t n_blocks = (n + B - 1) / B;

#pragma omp parallel
    {
        std::vector<float> cdist((size_t)8 * nc);
        std::vector<std::pair<float, int32_t>> cand(nc);
        std::vector<float> selbuf(nc);
        std::vector<uint64_t> pairs((size_t)B * P);
        std::vector<float> Qbuf, out, Qc;
        std::vector<int64_t> qlist;
        std::vector<float> hd((size_t)B * k);
        std::vector<int64_t> hid((size_t)B * k);
        std::vector<std::pair<double, int64_t>> fin(k);

        double st[5] = {0, 0, 0, 0, 0};
        auto now = [] { return std::chrono::steady_clock::now(); };
        auto since = [](auto a) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count(); };
#pragma omp for schedule(dynamic, 1)
        for (int64_t blk = 0; blk < n_blocks; ++blk) {
            auto T0 = now();
            const int64_t s0 = blk * B, s1 = std::min(n, s0 + B), q = s1 - s0;
            // Gather this block's query rows (sorted order).
            Qbuf.resize((size_t)q * d);
            for (int64_t i = 0; i < q; ++i)
                std::memcpy(&Qbuf[i * d], X + order[s0 + i] * d, d * sizeof(float));
            // 1. P nearest centroids per query (ties -> lower centroid id).
            for (int64_t i = 0; i < q; ++i) {
                if (i % 8 == 0) {  // centroid distances for the next 8 queries
                    const int64_t qq = std::min<int64_t>(8, q - i);
                    auto T2 = now();
                    block_dist(&Qbuf[i * d], qq, d, CT.data(), nc, nc, cn.data(), cdist.data());
                    st[4] += since(T2);
                }
                const float* crow = cdist.data() + (i % 8) * nc;
                // P nearest centroids, ties -> lower id: threshold by nth_element,
                // then one ordered scan.
                std::memcpy(selbuf.data(), crow, nc * sizeof(float));
                std::nth_element(selbuf.begin(), selbuf.begin() + (P - 1), selbuf.begin() + nc);
                const float thr = selbuf[P - 1];
                int64_t got = 0;
                for (int64_t j = 0; j < nc; ++j)
                    if (crow[j] < thr) cand[got++].second = (int32_t)j;
                for (int64_t j = 0; j < nc && got < P; ++j)
                    if (crow[j] == thr) cand[got++].second = (int32_t)j;
                for (int64_t j = 0; j < P; ++j)
                    pairs[i * P + j] = ((uint64_t)(uint32_t)cand[j].second << 32) | (uint64_t)i;
            }
            st[0] += since(T0); T0 = now();
            // 2. Scan each probed cluster once for all queries that need it.
            std::sort(pairs.begin(), pairs.begin() + q * P);
            for (int64_t i = 0; i < q * k; ++i) { hd[i] = kInf; hid[i] = -1; }
            int64_t a = 0;
            while (a < q * P) {
                const int64_t c = (int64_t)(pairs[a] >> 32);
                int64_t b = a;
                while (b < q * P && (int64_t)(pairs[b] >> 32) == c) ++b;
                const int64_t mq = b - a;
                const int64_t cs = offsets[c], m = offsets[c + 1] - cs;
                qlist.resize(mq);
                std::vector<float>& Qg = out;
                Qc.resize((size_t)mq * d);
                for (int64_t r = 0; r < mq; ++r) {
                    qlist[r] = (int64_t)(pairs[a + r] & 0xffffffffu);
                    std::memcpy(&Qc[r * d], &Qbuf[qlist[r] * d], d * sizeof(float));
                }
                Qg.resize((size_t)mq * m);
                auto T1 = now();
                block_dist(Qc.data(), mq, d, XT.data() + cs, n, m, xn.data() + cs, Qg.data());
                st[1] += since(T1); T1 = now();
                for (int64_t r = 0; r < mq; ++r) {
                    const int64_t i = qlist[r], self = s0 + i;
                    const float qn = xn[self];
                    Heap h{&hd[i * k], &hid[i * k], k};
                    const float* row = &Qg[r * m];
                    for (int64_t g0 = 0; g0 < m; g0 += 64) {
                        const int64_t g1 = std::min(m, g0 + 64);
                        float gmin = row[g0];
                        for (int64_t j = g0 + 1; j < g1; ++j) gmin = std::min(gmin, row[j]);
                        if (qn + gmin >= h.d[0]) continue;
                        for (int64_t j = g0; j < g1; ++j) {
                            const int64_t sp = cs + j;
                            if (sp == self) continue;
                            const float dist = qn + row[j];
                            if (dist < h.d[0]) h.replace_top(dist, sp);
                        }
                    }
                }
                st[2] += since(T1);
                a = b;
            }
            T0 = now();
            // 3. Exact distances for the winners, nearest first.
            for (int64_t i = 0; i < q; ++i) {
                const float* xq = &Qbuf[i * d];
                int64_t cnt = 0;
                for (int64_t j = 0; j < k; ++j) {
                    const int64_t sp = hid[i * k + j];
                    if (sp < 0) continue;
                    const float* xc = X + order[sp] * d;
                    double s = 0;
                    for (int64_t t = 0; t < d; ++t) {
                        const double df = (double)xq[t] - (double)xc[t];
                        s += df * df;
                    }
                    fin[cnt++] = {s, order[sp]};
                }
                std::sort(fin.begin(), fin.begin() + cnt);
                const int64_t orow = order[s0 + i];
                for (int64_t j = 0; j < cnt; ++j) {
                    res.indices[orow * k + j] = fin[j].second;
                    res.distances[orow * k + j] = (float)std::sqrt(fin[j].first);
                }
            }
            st[3] += since(T0);
        }
        if (dbg && omp_get_thread_num() == 0)
            std::fprintf(stderr, "[knn] t0: probe %.2f (centroid dist %.2f)  dist %.2f  heap %.2f  final %.2f\n", st[0], st[4], st[1], st[2], st[3]);
    }
    lap("search");
    (void)where;
    return res;
}

}  // namespace aof
