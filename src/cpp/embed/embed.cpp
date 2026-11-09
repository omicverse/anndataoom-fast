// C++ port of brisc's PaCMAP and LocalMAP embeddings (brisc/embed.pyx:
// `get_neighbor_pairs`, `sample_mid_near_pairs`, `sample_further_pairs`,
// `sample_further_pairs_nearby`, `reformat_for_parallel`, `get_gradient`,
// `get_scaled_gradient`, `update_embedding_adam_parallel`, `pacmap_parallel`,
// `localmap_parallel`; PCG RNG and heap helpers from brisc/cyutils.pxd).
//
// brisc (https://github.com/briscverse/brisc) is
//
//   MIT License
//
//   Copyright (c) 2026 Michael Wainberg and Keon Arbabi
//
//   Permission is hereby granted, free of charge, to any person obtaining a
//   copy of this software and associated documentation files (the
//   "Software"), to deal in the Software without restriction, including
//   without limitation the rights to use, copy, modify, merge, publish,
//   distribute, sublicense, and/or sell copies of the Software, and to permit
//   persons to whom the Software is furnished to do so, subject to the
//   following conditions:
//
//   The above copyright notice and this permission notice shall be included
//   in all copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
//   NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//   DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
//   OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
//   USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// This port reproduces brisc's *parallel* code path (what brisc computes with
// num_threads > 1, or num_threads=1 with match_parallel=True), whose result
// does not depend on the number of threads.  brisc's extension is built with
// -Ofast, so its arithmetic is not the literal Cython source: GCC fuses
// multiply-adds, reassociates constants, vectorises the PC-distance
// reduction, replaces `pow` by `expf(t * log(beta))` and `x / sqrt(d)` by an
// RSQRTSS estimate plus one Newton step.  The kernels in embed_kernels.inc
// spell out exactly that sequence of IEEE single-precision operations, with
// explicit fused multiply-adds, so this file is compiled without fast-math
// and without FMA contraction and gives the same bits on every compiler.
// The one hardware-defined operation is the RSQRTSS estimate used in
// LocalMAP's third phase; on non-x86 CPUs it is replaced by 1 / sqrt(d).
//
// This port is part of anndataoom-fast and is distributed under the GNU
// General Public License, version 3 or later.

#include "embed.hpp"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define AOF_EMBED_X86 1
#include <xmmintrin.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define AOF_PREFETCH(p) __builtin_prefetch((const void*)(p))
#define AOF_FMAF(a, b, c) __builtin_fmaf((a), (b), (c))
#define AOF_SQRTF(x) __builtin_sqrtf(x)
#elif defined(AOF_EMBED_X86)
#define AOF_PREFETCH(p) _mm_prefetch((const char*)(p), _MM_HINT_T0)
#define AOF_FMAF(a, b, c) std::fma((float)(a), (float)(b), (float)(c))
#define AOF_SQRTF(x) std::sqrt((float)(x))
#else
#define AOF_PREFETCH(p) ((void)(p))
#define AOF_FMAF(a, b, c) std::fma((float)(a), (float)(b), (float)(c))
#define AOF_SQRTF(x) std::sqrt((float)(x))
#endif

namespace aof_embed {

namespace {

// ---------------------------------------------------------------------------
// PCG RNG (brisc/cyutils.pxd `rand`, `randint`, `srand`)
// ---------------------------------------------------------------------------
inline uint32_t pcg_rand(uint64_t& state) {
    const uint64_t x = state;
    state = x * 6364136223846793005ULL + 1442695040888963407ULL;
    const uint32_t s = uint32_t((x ^ (x >> 18)) >> 27);
    const uint32_t rot = uint32_t(x >> 59);
    return (s >> rot) | (s << ((0u - rot) & 31u));
}

// Lemire's method
inline uint32_t pcg_randint(uint32_t bound, uint64_t& state) {
    uint64_t m = uint64_t(pcg_rand(state)) * uint64_t(bound);
    uint32_t l = uint32_t(m);
    if (l < bound) {
        const uint32_t threshold = (0u - bound) % bound;
        while (l < threshold) {
            m = uint64_t(pcg_rand(state)) * uint64_t(bound);
            l = uint32_t(m);
        }
    }
    return uint32_t(m >> 32);
}

inline uint64_t pcg_srand(uint64_t seed) {
    uint64_t state = seed + 1442695040888963407ULL;
    pcg_rand(state);
    return state;
}

// ---------------------------------------------------------------------------
// Max-heap helpers (brisc/cyutils.pxd), 1-based internally
// ---------------------------------------------------------------------------
inline void max_heap_pop(uint32_t* labels, float* dists, uint32_t k) {
    uint32_t j = 1;
    labels -= 1;
    dists -= 1;
    const float distance = dists[k];
    const uint32_t label = labels[k];
    for (;;) {
        uint32_t child = j << 1;
        if (child > k) break;
        child += (child < k && dists[child] <= dists[child + 1]) ? 1u : 0u;
        if (distance > dists[child]) break;
        dists[j] = dists[child];
        labels[j] = labels[child];
        j = child;
    }
    dists[j] = distance;
    labels[j] = label;
}

inline void max_heap_replace_top(uint32_t* labels, float* dists, uint32_t label, float distance,
                                 uint32_t k) {
    uint32_t j = 1;
    labels -= 1;
    dists -= 1;
    for (;;) {
        uint32_t child = j << 1;
        if (child > k) break;
        child += (child < k && dists[child] <= dists[child + 1]) ? 1u : 0u;
        if (distance > dists[child]) break;
        dists[j] = dists[child];
        labels[j] = labels[child];
        j = child;
    }
    dists[j] = distance;
    labels[j] = label;
}

inline void max_heap_sort(uint32_t* labels, float* dists, uint32_t k) {
    for (uint32_t j = 0; j < k; j++) {
        const float distance = dists[0];
        const uint32_t label = labels[0];
        max_heap_pop(labels, dists, k - j);
        dists[k - j - 1] = distance;
        labels[k - j - 1] = label;
    }
}

// ---------------------------------------------------------------------------
// Floating-point helpers shared by both kernel builds
// ---------------------------------------------------------------------------

// Starting value of the reciprocal square root refined by one Newton step in
// LocalMAP phase 3.  brisc's compiler emits the CPU's RSQRTSS estimate, whose
// bits differ between CPU vendors; by default we start from the correctly
// rounded 1 / sqrt(x) instead, so the embedding is identical on every
// machine.  AOF_EMBED_BRISC_RSQRT=1 (x86 only) uses RSQRTSS and reproduces
// brisc bit for bit on the same CPU.
bool brisc_rsqrt_enabled() {
#if defined(AOF_EMBED_X86)
    static const bool on = [] {
        const char* v = std::getenv("AOF_EMBED_BRISC_RSQRT");
        return v && v[0] && v[0] != '0';
    }();
    return on;
#else
    return false;
#endif
}

inline float rsqrt_estimate(float x) {
#if defined(AOF_EMBED_X86)
    static const bool hw = brisc_rsqrt_enabled();
    if (hw) return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x)));
#endif
    return 1.0f / std::sqrt(x);
}

// 8 -> 4 -> 2 -> 1 lane fold of GCC's vector reduction epilogue
inline float hsum8(const float* v) {
    float v4[4], v2[2];
    for (int l = 0; l < 4; l++) v4[l] = v[l + 4] + v[l];
    for (int l = 0; l < 2; l++) v2[l] = v4[l + 2] + v4[l];
    return v2[1] + v2[0];
}

// float from its IEEE bit pattern
inline float bits_to_float(uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

// Adam bias correction, as brisc computes it: GCC turns
// pow(beta, t) into expf(t * logf(beta)) with logf(beta) folded to a float
// constant.  (float)exp((double)x) equals glibc's expf on every argument used
// here (t <= 200000: the exact value is never within 4e-13 relative of a float
// rounding boundary), so any libm gives the same bits.
inline float adam_learning_rate(float learning_rate, uint32_t iteration) {
    static const float log_beta2 = bits_to_float(0xba8322caU);  // logf(0.999f)
    static const float log_beta1 = bits_to_float(0xbdd7c745U);  // logf(0.9f)
    const float t = float(iteration + 1u);
    const float p2 = float(std::exp(double(t * log_beta2)));
    const float p1 = float(std::exp(double(t * log_beta1)));
    const float a = std::sqrt(1.0f - p2);
    return (a / (1.0f - p1)) * learning_rate;
}

// One optimisation step for a range of cells: gradient (brisc `get_gradient`
// / `get_scaled_gradient`) from `embedding`, then the Adam update into
// `emb_out`.
struct GradArgs {
    const float* embedding;
    float* emb_out;
    float* momentum;
    float* velocity;
    const uint64_t *nn_ptr, *mn_ptr, *fp_ptr;
    const uint32_t *nn_idx, *mn_idx, *fp_idx;
    float w_neighbors, w_mid_near, half_max_distance;
    float one_minus_beta1, one_minus_beta2, learning_rate;
};

struct MidNearArgs {
    const float* PCs;
    uint64_t n, d;
    uint32_t* mid_near_pairs;
    uint32_t num_mid_near_pairs;
    uint64_t seed;
};

struct NearbyArgs {
    const float* embedding;
    uint64_t n;
    const uint32_t* neighbor_pairs;
    uint32_t* further_pairs;
    uint32_t num_neighbors, num_further_pairs;
    uint64_t seed;
    float max_distance_squared;
};

// Portable kernels
#define AOF_KATTR
#define AOF_KNAME(name) name##_generic
#include "embed_kernels.inc"
#undef AOF_KATTR
#undef AOF_KNAME

// x86 kernels with hardware FMA (identical results, faster)
#if defined(AOF_EMBED_X86) && (defined(__GNUC__) || defined(__clang__))
#define AOF_EMBED_FMA_KERNELS 1
#define AOF_KATTR __attribute__((target("avx2,fma")))
#define AOF_KNAME(name) name##_fma
#include "embed_kernels.inc"
#undef AOF_KATTR
#undef AOF_KNAME
#endif

struct Kernels {
    void (*gradients)(const GradArgs&, int64_t, int64_t);
    void (*scaled_gradients)(const GradArgs&, int64_t, int64_t);
    void (*mid_near)(const MidNearArgs&, int64_t, int64_t);
    void (*further_nearby)(const NearbyArgs&, int64_t, int64_t);
};

Kernels pick_kernels() {
#if defined(AOF_EMBED_FMA_KERNELS)
    // AOF_EMBED_GENERIC=1 forces the portable kernels (used by the tests to
    // check that both builds agree bit for bit).
    const char* force = std::getenv("AOF_EMBED_GENERIC");
    const bool generic = force && force[0] && force[0] != '0';
    __builtin_cpu_init();
    if (!generic && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        return {gradients_fma, scaled_gradients_fma, mid_near_fma, further_nearby_fma};
#endif
    return {gradients_generic, scaled_gradients_generic, mid_near_generic,
            further_nearby_generic};
}

const Kernels& kernels() {
    static const Kernels k = pick_kernels();
    return k;
}

// Run f(begin, end) over [0, n) in fixed blocks, in parallel.  Every kernel
// writes only its own block's outputs, so the result is thread-independent.
template <class F>
void parallel_blocks(int64_t n, int64_t block, int nt, F&& f) {
    const int64_t nb = (n + block - 1) / block;
    if (nb <= 0) return;
    // No more threads than blocks: idle threads would only add barrier cost.
    nt = int(std::min<int64_t>(nt, nb));
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
    for (int64_t b = 0; b < nb; b++) {
        const int64_t begin = b * block;
        f(begin, std::min(n, begin + block));
    }
}

// ---------------------------------------------------------------------------
// get_neighbor_pairs
// ---------------------------------------------------------------------------
void neighbor_pairs(const uint32_t* knn, const float* knn_d, uint64_t n, uint64_t k,
                    uint32_t num_neighbors, uint32_t* pairs, int nt) {
    std::vector<float> total(n);
    const int64_t N = int64_t(n);
    nt = int(std::min<int64_t>(nt, N / 4096 + 1));
#pragma omp parallel for schedule(static) num_threads(nt)
    for (int64_t i = 0; i < N; i++) {
        const float* d = knn_d + uint64_t(i) * k;
        float t = std::sqrt(d[3]) + std::sqrt(d[4]);
        t = t + std::sqrt(d[5]);
        if (t < 3e-10f) t = 3e-10f;
        total[size_t(i)] = t;
    }
    std::atomic<int> too_large{0}, self{0};
#pragma omp parallel num_threads(nt)
    {
        std::vector<float> heap(num_neighbors);
#pragma omp for schedule(static)
        for (int64_t ii = 0; ii < N; ii++) {
            const uint64_t i = uint64_t(ii);
            uint32_t* lab = pairs + i * num_neighbors;
            for (uint32_t j = 0; j < num_neighbors; j++) {
                heap[j] = FLT_MAX;
                lab[j] = 0;
            }
            float worst = FLT_MAX;
            for (uint64_t j = 0; j < k; j++) {
                const uint32_t nb = knn[i * k + j];
                if (nb >= n) {
                    too_large.store(1, std::memory_order_relaxed);
                    break;
                }
                if (nb == i) {
                    self.store(1, std::memory_order_relaxed);
                    break;
                }
                const float sd = knn_d[i * k + j] / total[nb];
                if (sd < worst) {
                    max_heap_replace_top(lab, heap.data(), nb, sd, num_neighbors);
                    worst = heap[0];
                }
            }
            max_heap_sort(lab, heap.data(), num_neighbors);
        }
    }
    if (too_large.load())
        throw std::invalid_argument(
            "some nearest-neighbor indices are >= the number of cells");
    if (self.load())
        throw std::invalid_argument(
            "some nearest-neighbor indices indicate that a cell is its own neighbor "
            "(knn[i, j] == i); pass neighbours excluding the cell itself");
}

// ---------------------------------------------------------------------------
// sample_further_pairs (initial)
// ---------------------------------------------------------------------------
void further_pairs_initial(const uint32_t* nb_pairs, uint32_t nnb, uint64_t n, uint32_t nfp,
                           uint64_t seed, uint32_t* fp, int nt) {
    const int64_t N = int64_t(n);
    const uint32_t n32 = uint32_t(n);
    nt = int(std::min<int64_t>(nt, N / 1024 + 1));
#pragma omp parallel for schedule(dynamic, 1024) num_threads(nt)
    for (int64_t ii = 0; ii < N; ii++) {
        const uint32_t i = uint32_t(ii);
        uint64_t state = pcg_srand(seed + i);
        const uint32_t* nb = nb_pairs + uint64_t(i) * nnb;
        uint32_t* out = fp + uint64_t(i) * nfp;
        for (uint32_t j = 0; j < nfp; j++) {
            for (;;) {
                const uint32_t s = pcg_randint(n32, state);
                if (s == i) continue;
                bool dup = false;
                for (uint32_t q = 0; q < nnb && !dup; q++) dup = s == nb[q];
                for (uint32_t q = 0; q < j && !dup; q++) dup = s == out[q];
                if (dup) continue;
                out[j] = s;
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// reformat_for_parallel: pairs[i] (n x p) -> symmetric adjacency lists
// indices[indptr[c]:indptr[c+1]], each cell's list in the order obtained by
// scanning `pairs` row by row and appending j to i's list and i to j's list.
// (Cell c's own row lands after the reverse pairs (i, c) with i < c and
// before those with i > c.)  Same algorithm as brisc: bucket the reverse
// pairs by the thread range containing their target, keeping source order.
// ---------------------------------------------------------------------------
struct Adjacency {
    std::vector<uint64_t> indptr;
    std::vector<uint32_t> indices;
};

void reformat(const uint32_t* pairs, uint64_t n, uint32_t p, Adjacency& adj, int nt) {
    // The layout does not depend on T; use about one range per 16k cells.
    const uint64_t T = std::max<uint64_t>(1, std::min<uint64_t>(uint64_t(nt), n / 16384 + 1));
    const uint64_t chunk = (n + T - 1) / T;
    const uint64_t stride = T + 16;
    adj.indptr.assign(n + 1, 0);
    adj.indices.resize(2 * n * uint64_t(p));
    std::vector<uint32_t> bucket_src(n * uint64_t(p)), bucket_dst(n * uint64_t(p));
    std::vector<uint64_t> cursors(T * stride, 0), range_offsets(T + 1, 0);
    std::vector<uint64_t> dest(n);
    const int64_t TT = int64_t(T);
#pragma omp parallel for schedule(static, 1) num_threads(int(T))
    for (int64_t t = 0; t < TT; t++) {
        uint64_t* cur = cursors.data() + uint64_t(t) * stride;
        const uint64_t start = std::min(uint64_t(t) * chunk, n), end = std::min(start + chunk, n);
        for (uint64_t i = start; i < end; i++)
            for (uint32_t q = 0; q < p; q++) cur[pairs[i * p + q] / chunk]++;
    }
    uint64_t total = 0;
    for (uint64_t r = 0; r < T; r++) {
        range_offsets[r] = total;
        for (uint64_t t = 0; t < T; t++) {
            const uint64_t c = cursors[t * stride + r];
            cursors[t * stride + r] = total;
            total += c;
        }
    }
    range_offsets[T] = total;
#pragma omp parallel for schedule(static, 1) num_threads(int(T))
    for (int64_t t = 0; t < TT; t++) {
        uint64_t* cur = cursors.data() + uint64_t(t) * stride;
        const uint64_t start = std::min(uint64_t(t) * chunk, n), end = std::min(start + chunk, n);
        for (uint64_t i = start; i < end; i++)
            for (uint32_t q = 0; q < p; q++) {
                const uint32_t j = pairs[i * p + q];
                const uint64_t pos = cur[j / chunk]++;
                bucket_src[pos] = uint32_t(i);
                bucket_dst[pos] = j;
            }
    }
    uint64_t* indptr = adj.indptr.data();
#pragma omp parallel for schedule(static, 1) num_threads(int(T))
    for (int64_t r = 0; r < TT; r++) {
        const uint64_t start = std::min(uint64_t(r) * chunk, n), end = std::min(start + chunk, n);
        for (uint64_t i = start; i < end; i++) indptr[i + 1] = p;
        for (uint64_t q = range_offsets[r]; q < range_offsets[r + 1]; q++) indptr[bucket_dst[q] + 1]++;
    }
    for (uint64_t i = 2; i <= n; i++) indptr[i] += indptr[i - 1];
    uint32_t* idx = adj.indices.data();
#pragma omp parallel for schedule(static, 1) num_threads(int(T))
    for (int64_t r = 0; r < TT; r++) {
        const uint64_t start = std::min(uint64_t(r) * chunk, n), end = std::min(start + chunk, n);
        for (uint64_t i = start; i < end; i++) dest[i] = indptr[i];
        uint64_t own = start;
        for (uint64_t q = range_offsets[r]; q < range_offsets[r + 1]; q++) {
            const uint64_t i = bucket_src[q];
            while (own < end && own <= i) {
                for (uint32_t s = 0; s < p; s++) idx[dest[own]++] = pairs[own * p + s];
                own++;
            }
            const uint32_t j = bucket_dst[q];
            idx[dest[j]++] = uint32_t(i);
        }
        while (own < end) {
            for (uint32_t s = 0; s < p; s++) idx[dest[own]++] = pairs[own * p + s];
            own++;
        }
    }
}

}  // namespace

void embed(const float* PCs, uint64_t n, uint64_t d, const uint32_t* knn,
           const float* knn_sq_dist, uint64_t k, const Params& P, float* embedding,
           const PairsOut& pairs_out) {
    if (n >= uint64_t(UINT32_MAX)) throw std::invalid_argument("too many cells (must be < 2^32 - 1)");
    if (d < 2) throw std::invalid_argument("need at least 2 input dimensions");
    if (k < 6) throw std::invalid_argument("need at least 6 nearest neighbours per cell");
    if (P.num_neighbors < 1 || P.num_neighbors > k)
        throw std::invalid_argument("num_neighbors must be between 1 and the number of kNN columns");
    if (P.num_mid_near_pairs < 1 || P.num_further_pairs < 1)
        throw std::invalid_argument("num_mid_near_pairs and num_further_pairs must be >= 1");
    if (n < uint64_t(P.num_mid_near_pairs) + 6)
        throw std::invalid_argument("fewer than num_mid_near_pairs + 6 cells");
    if (n < uint64_t(P.num_neighbors) + P.num_further_pairs + 1)
        throw std::invalid_argument("fewer than num_neighbors + num_further_pairs + 1 cells");
    if (k >= n) throw std::invalid_argument("number of kNN columns must be < the number of cells");
    if (P.num_phase_1_iterations < 1 || P.num_phase_2_iterations < 1 || P.num_phase_3_iterations < 1)
        throw std::invalid_argument("iteration counts must be >= 1");

    const int nt = P.num_threads > 0 ? P.num_threads : omp_get_max_threads();
    const Kernels& K = kernels();
    const uint32_t nnb = P.num_neighbors, nmn = P.num_mid_near_pairs, nfp = P.num_further_pairs;
    const int64_t N = int64_t(n);

    // Pairs
    std::vector<uint32_t> nb_pairs(n * nnb), mn_pairs(n * nmn), fp_pairs(n * nfp);
    neighbor_pairs(knn, knn_sq_dist, n, k, nnb, nb_pairs.data(), nt);
    {
        MidNearArgs a{PCs, n, d, mn_pairs.data(), nmn, P.seed};
        parallel_blocks(N, 256, nt, [&](int64_t b, int64_t e) { K.mid_near(a, b, e); });
    }
    // brisc: seed + num_cells * num_mid_near_pairs, in 32-bit unsigned arithmetic
    const uint64_t fp_seed = P.seed + uint64_t(uint32_t(uint32_t(n) * nmn));
    further_pairs_initial(nb_pairs.data(), nnb, n, nfp, fp_seed, fp_pairs.data(), nt);

    Adjacency nb_adj, mn_adj, fp_adj;
    reformat(nb_pairs.data(), n, nnb, nb_adj, nt);
    reformat(mn_pairs.data(), n, nmn, mn_adj, nt);
    reformat(fp_pairs.data(), n, nfp, fp_adj, nt);

    // Optimisation (embedding double-buffered: emb[cur] -> emb[1 - cur])
    std::vector<float> buffer(2 * n), momentum(2 * n), velocity(2 * n);
    float* emb[2] = {embedding, buffer.data()};
#pragma omp parallel for schedule(static) num_threads(int(std::min<int64_t>(nt, N / 4096 + 1)))
    for (int64_t i = 0; i < N; i++) {
        embedding[2 * i] = 0.01f * PCs[uint64_t(i) * d];
        embedding[2 * i + 1] = 0.01f * PCs[uint64_t(i) * d + 1];
        momentum[2 * i] = momentum[2 * i + 1] = 0.0f;
        velocity[2 * i] = velocity[2 * i + 1] = 0.0f;
    }
    int cur = 0;
    const uint32_t it1 = P.num_phase_1_iterations, it12 = it1 + P.num_phase_2_iterations,
                   num_iterations = it12 + P.num_phase_3_iterations;
    const float inv_it1 = 1.0f / float(it1);
    const float beta1 = 0.9f, beta2 = 0.999f;
    const float half_max_distance = 0.5f * P.max_distance;
    const float max_distance_squared = P.max_distance * P.max_distance;
    float w_mid_near = 0.0f, w_neighbors = 0.0f;
    GradArgs ga{};
    ga.momentum = momentum.data();
    ga.velocity = velocity.data();
    ga.half_max_distance = half_max_distance;
    ga.one_minus_beta1 = 1.0f - beta1;
    ga.one_minus_beta2 = 1.0f - beta2;
    const int64_t block = 2048;
    for (uint32_t it = 0; it < num_iterations; it++) {
        if (it < it1) {
            const float frac = float(it) * inv_it1;
            w_mid_near = AOF_FMAF(1.0f - frac, 1000.0f, frac * 3.0f);
            w_neighbors = 2.0f;
        } else if (it < it12) {
            w_mid_near = 3.0f;
            w_neighbors = 3.0f;
        } else if (!P.localmap) {
            w_mid_near = 0.0f;
            w_neighbors = 1.0f;
        }
        ga.embedding = emb[cur];
        ga.emb_out = emb[1 - cur];
        ga.nn_ptr = nb_adj.indptr.data();
        ga.nn_idx = nb_adj.indices.data();
        ga.mn_ptr = mn_adj.indptr.data();
        ga.mn_idx = mn_adj.indices.data();
        ga.fp_ptr = fp_adj.indptr.data();
        ga.fp_idx = fp_adj.indices.data();
        ga.w_neighbors = w_neighbors;
        ga.w_mid_near = w_mid_near;
        ga.learning_rate = adam_learning_rate(P.learning_rate, it);
        if (P.localmap && it >= it12)
            parallel_blocks(N, block, nt, [&](int64_t b, int64_t e) { K.scaled_gradients(ga, b, e); });
        else
            parallel_blocks(N, block, nt, [&](int64_t b, int64_t e) { K.gradients(ga, b, e); });
        cur = 1 - cur;

        // LocalMAP: re-sample further pairs every 10 iterations in phase 3
        if (P.localmap && it >= it12 && it % 10 == 0) {
            NearbyArgs na{emb[cur], n, nb_pairs.data(), fp_pairs.data(), nnb, nfp, P.seed + it,
                          max_distance_squared};
            parallel_blocks(N, 1024, nt, [&](int64_t b, int64_t e) { K.further_nearby(na, b, e); });
            reformat(fp_pairs.data(), n, nfp, fp_adj, nt);
        }
    }
    if (emb[cur] != embedding) std::copy(emb[cur], emb[cur] + 2 * n, embedding);

    if (pairs_out.neighbor_pairs)
        std::copy(nb_pairs.begin(), nb_pairs.end(), pairs_out.neighbor_pairs);
    if (pairs_out.mid_near_pairs)
        std::copy(mn_pairs.begin(), mn_pairs.end(), pairs_out.mid_near_pairs);
    if (pairs_out.further_pairs)
        std::copy(fp_pairs.begin(), fp_pairs.end(), pairs_out.further_pairs);
}

}  // namespace aof_embed
