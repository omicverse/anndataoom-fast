#include "umap.hpp"

#include <omp.h>

#include <cmath>
#include <cstring>
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
#pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < E; ++i)
                update_edge(i, ep, emb, dim, n, head, tail, eps, st, rng.data(), p, alpha);
        } else {
            for (int64_t i = 0; i < E; ++i)
                update_edge(i, ep, emb, dim, n, head, tail, eps, st, rng.data(), p, alpha);
        }
        alpha = p.initial_alpha * (1.0 - ((double)ep / (double)p.n_epochs));
    }
}

void csr_matvec(const int64_t* indptr, const int32_t* indices, const double* data, int64_t n,
                const double* x, double* y) {
#pragma omp parallel for schedule(static, 2048)
    for (int64_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (int64_t e = indptr[i]; e < indptr[i + 1]; ++e) s += data[e] * x[indices[e]];
        y[i] = s;
    }
}

}  // namespace aof
