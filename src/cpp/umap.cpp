#include "umap.hpp"

#include <omp.h>

#include <cmath>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace aof {

namespace {

// umap.utils.tau_rand_int, with numba's int64 semantics: left shifts wrap,
// right shifts are arithmetic, the result is truncated to int32.
inline int32_t tau_rand_int(int64_t* s) {
    auto shl = [](int64_t v, int k) { return (int64_t)((uint64_t)v << k); };
    s[0] = (shl(s[0] & 4294967294LL, 12) & 0xFFFFFFFFLL) ^ (((shl(s[0], 13) & 0xFFFFFFFFLL) ^ s[0]) >> 19);
    s[1] = (shl(s[1] & 4294967288LL, 4) & 0xFFFFFFFFLL) ^ (((shl(s[1], 2) & 0xFFFFFFFFLL) ^ s[1]) >> 25);
    s[2] = (shl(s[2] & 4294967280LL, 17) & 0xFFFFFFFFLL) ^ (((shl(s[2], 3) & 0xFFFFFFFFLL) ^ s[2]) >> 11);
    return (int32_t)(s[0] ^ s[1] ^ s[2]);
}

inline double clip4(double v) { return v > 4.0 ? 4.0 : (v < -4.0 ? -4.0 : v); }

inline float rdist(const float* x, const float* y, int dim) {
    float r = 0.f;
    for (int d = 0; d < dim; ++d) {
        const float diff = x[d] - y[d];
        r += diff * diff;
    }
    return r;
}

inline int64_t pymod(int64_t v, int64_t m) {
    int64_t r = v % m;
    return r < 0 ? r + m : r;
}

struct EdgeState {
    std::vector<double> next_sample, next_neg, per_neg;
};

// One edge update (attractive move + negative samples): umap's
// _optimize_layout_euclidean_single_epoch body in strict IEEE arithmetic.
// umap-learn compiles the same loop with numba fastmath (reassociation,
// reciprocal division, FMA contraction on capable CPUs), so its output is
// itself CPU/LLVM dependent; this version is reproducible everywhere and
// matches a plain-IEEE evaluation of umap's code exactly.
inline void update_edge(int64_t i, int64_t n_epoch, float* emb, int dim, int64_t n, const int32_t* head,
                        const int32_t* tail, const double* eps, EdgeState& st, int64_t* rng,
                        const UmapSgdParams& p, double alpha) {
    if (!(st.next_sample[i] <= (double)n_epoch)) return;
    const int64_t j = head[i];
    int64_t k = tail[i];
    float* current = emb + j * dim;
    float* other = emb + k * dim;
    float dist_squared = rdist(current, other, dim);
    double grad_coeff;
    if (dist_squared > 0.f) {
        grad_coeff = -2.0 * p.a * p.b * std::pow((double)dist_squared, p.b - 1.0);
        grad_coeff /= p.a * std::pow((double)dist_squared, p.b) + 1.0;
    } else {
        grad_coeff = 0.0;
    }
    for (int d = 0; d < dim; ++d) {
        const double grad_d = clip4(grad_coeff * (double)(current[d] - other[d]));
        current[d] = (float)((double)current[d] + grad_d * alpha);
        if (p.move_other) other[d] = (float)((double)other[d] + (-grad_d) * alpha);
    }
    st.next_sample[i] += eps[i];
    const int64_t n_neg = (int64_t)(((double)n_epoch - st.next_neg[i]) / st.per_neg[i]);
    int64_t* rs = rng + j * 3;
    for (int64_t q = 0; q < n_neg; ++q) {
        k = pymod((int64_t)tau_rand_int(rs), n);
        other = emb + k * dim;
        dist_squared = rdist(current, other, dim);
        if (dist_squared > 0.f) {
            grad_coeff = 2.0 * p.gamma * p.b;
            grad_coeff /= (0.001 + (double)dist_squared) * (p.a * std::pow((double)dist_squared, p.b) + 1.0);
        } else if (j == k) {
            continue;
        } else {
            grad_coeff = 0.0;
        }
        for (int d = 0; d < dim; ++d) {
            double grad_d = 0.0;
            if (grad_coeff > 0.0) grad_d = clip4(grad_coeff * (double)(current[d] - other[d]));
            current[d] = (float)((double)current[d] + grad_d * alpha);
        }
    }
    st.next_neg[i] += (double)n_neg * st.per_neg[i];
}

}  // namespace

void umap_optimize_euclidean(float* emb, int64_t n, int dim, const int32_t* head, const int32_t* tail,
                             int64_t E, const double* eps, const int64_t* rng_state,
                             const UmapSgdParams& p) {
    EdgeState st;
    st.per_neg.resize(E);
    for (int64_t i = 0; i < E; ++i) st.per_neg[i] = eps[i] / p.negative_sample_rate;
    st.next_neg = st.per_neg;
    st.next_sample.assign(eps, eps + E);
    // rng_state_per_sample = rng_state + view(float64(embedding[:, 0]), int64)
    std::vector<int64_t> rng((size_t)n * 3);
    for (int64_t v = 0; v < n; ++v) {
        const double x0 = (double)emb[v * dim];
        int64_t bits;
        std::memcpy(&bits, &x0, sizeof(bits));
        for (int c = 0; c < 3; ++c) rng[v * 3 + c] = (int64_t)((uint64_t)rng_state[c] + (uint64_t)bits);
    }
    double alpha = p.initial_alpha;
    for (int ep = 0; ep < p.n_epochs; ++ep) {
        if (p.parallel) {
            const int nt = (int)std::max<int64_t>(
                1, std::min<int64_t>(omp_get_max_threads(), (E + 4095) / 4096));
#pragma omp parallel for schedule(static) num_threads(nt)
            for (int64_t i = 0; i < E; ++i)
                update_edge(i, ep, emb, dim, n, head, tail, eps, st, rng.data(), p, alpha);
        } else {
            for (int64_t i = 0; i < E; ++i)
                update_edge(i, ep, emb, dim, n, head, tail, eps, st, rng.data(), p, alpha);
        }
        alpha = p.initial_alpha * (1.0 - ((double)ep / (double)p.n_epochs));
    }
}

namespace {

inline uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// One pow per term: d2^b = d2 * d2^(b-1).
inline double attract_coeff(float d2, const UmapSgdParams& p) {
    if (!(d2 > 0.f)) return 0.0;
    const double t = std::pow((double)d2, p.b - 1.0);
    return -2.0 * p.a * p.b * t / (p.a * t * (double)d2 + 1.0);
}

inline double repulse_coeff(float d2, const UmapSgdParams& p) {
    return 2.0 * p.gamma * p.b / ((0.001 + (double)d2) * (p.a * std::pow((double)d2, p.b) + 1.0));
}

}  // namespace

void umap_optimize_euclidean_det(float* emb, int64_t n, int dim, const int64_t* indptr,
                                 const int32_t* tail, const double* eps, uint64_t seed,
                                 const UmapSgdParams& p, int64_t block) {
    if (dim > 64) throw std::invalid_argument("dim must be <= 64");
    // Blocks must be small relative to n (a vertex rarely shares a block with
    // its neighbours); the size depends on n only, never on the thread count.
    if (block <= 0) block = std::min<int64_t>(4096, std::max<int64_t>(256, n / 64));
    const int64_t E = indptr[n];
    std::vector<double> next_sample(eps, eps + E), per_neg(E), next_neg(E);
    for (int64_t e = 0; e < E; ++e) next_neg[e] = per_neg[e] = eps[e] / p.negative_sample_rate;
    int64_t max_block_edges = 0;
    for (int64_t b0 = 0; b0 < n; b0 += block)
        max_block_edges = std::max(max_block_edges, indptr[std::min(n, b0 + block)] - indptr[b0]);
    std::vector<float> out((size_t)block * dim);
    std::vector<unsigned char> moved((size_t)block);
    // Tail moves of the block's edges (umap's move_other), applied after the block.
    std::vector<float> tail_delta((size_t)max_block_edges * dim);
    std::vector<unsigned char> sampled((size_t)max_block_edges);
    double alpha = p.initial_alpha;
    for (int ep = 0; ep < p.n_epochs; ++ep) {
        // Vertices are visited in fixed blocks, in index order -- umap's own
        // edge order, since its edges are sorted by head.  Inside a block each
        // head vertex is updated by one thread from the live coordinates; the
        // moves umap applies to the tail of each edge are recorded and added
        // after the block, in edge order.  No read races with a write, so the
        // result is independent of the number of threads.
        for (int64_t b0 = 0; b0 < n; b0 += block) {
            const int64_t b1 = std::min<int64_t>(n, b0 + block);
            const int64_t e0 = indptr[b0];
            // One thread per 64 vertices of the block at most: a small block on
            // many threads spends its time in the barrier (and stalls when the
            // node is oversubscribed).  The result does not depend on it.
            const int nt = (int)std::max<int64_t>(
                1, std::min<int64_t>(omp_get_max_threads(), (b1 - b0 + 63) / 64));
#pragma omp parallel for schedule(dynamic, 64) num_threads(nt)
            for (int64_t v = b0; v < b1; ++v) {
                float cur[64];
                const float* self = emb + v * dim;
                for (int d = 0; d < dim; ++d) cur[d] = self[d];
                bool mv = false;
                for (int64_t e = indptr[v]; e < indptr[v + 1]; ++e) {
                    if (!(next_sample[e] <= (double)ep)) {
                        sampled[e - e0] = 0;
                        continue;
                    }
                    mv = true;
                    sampled[e - e0] = p.move_other;
                    const float* oth = emb + (int64_t)tail[e] * dim;
                    float* td = &tail_delta[(size_t)(e - e0) * dim];
                    const double gc = attract_coeff(rdist(cur, oth, dim), p);
                    for (int d = 0; d < dim; ++d) {
                        const double g = clip4(gc * (double)(cur[d] - oth[d]));
                        cur[d] = (float)((double)cur[d] + g * alpha);
                        td[d] = (float)(-g * alpha);
                    }
                    next_sample[e] += eps[e];
                    const int64_t n_neg = (int64_t)(((double)ep - next_neg[e]) / per_neg[e]);
                    uint64_t st = seed ^ ((uint64_t)e * 0xD1B54A32D192ED03ULL) ^
                                  ((uint64_t)ep * 0x8CB92BA72F3D8DD7ULL);
                    for (int64_t q = 0; q < n_neg; ++q) {
                        const uint64_t r = splitmix64(st);
                        const int64_t k = (int64_t)(((r >> 32) * (uint64_t)n) >> 32);
                        if (k == v) continue;
                        const float* o2 = emb + k * dim;
                        const float d2 = rdist(cur, o2, dim);
                        if (!(d2 > 0.f)) continue;
                        const double gr = repulse_coeff(d2, p);
                        for (int d = 0; d < dim; ++d) {
                            const double g = clip4(gr * (double)(cur[d] - o2[d]));
                            cur[d] = (float)((double)cur[d] + g * alpha);
                        }
                    }
                    next_neg[e] += (double)n_neg * per_neg[e];
                }
                moved[v - b0] = mv;
                if (mv) for (int d = 0; d < dim; ++d) out[(v - b0) * dim + d] = cur[d];
            }
            for (int64_t v = b0; v < b1; ++v)
                if (moved[v - b0])
                    std::memcpy(emb + v * dim, &out[(v - b0) * dim], sizeof(float) * dim);
            const int64_t e1 = indptr[b1];
            for (int64_t e = e0; e < e1; ++e) {
                if (!sampled[e - e0]) continue;
                float* o = emb + (int64_t)tail[e] * dim;
                const float* td = &tail_delta[(size_t)(e - e0) * dim];
                for (int d = 0; d < dim; ++d) o[d] = (float)((double)o[d] + (double)td[d]);
            }
        }
        alpha = p.initial_alpha * (1.0 - ((double)ep / (double)p.n_epochs));
    }
}

void csr_matvec(const int64_t* indptr, const int32_t* indices, const double* data, int64_t n,
                const double* x, double* y) {
    // Small graphs: few threads (each wake-up costs more than the rows it does).
    const int nt = (int)std::max<int64_t>(
        1, std::min<int64_t>(omp_get_max_threads(), (n + 2047) / 2048));
#pragma omp parallel for schedule(static, 2048) num_threads(nt)
    for (int64_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (int64_t e = indptr[i]; e < indptr[i + 1]; ++e) s += data[e] * x[indices[e]];
        y[i] = s;
    }
}

}  // namespace aof
