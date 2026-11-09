// UMAP layout optimisation: a port of umap-learn 0.5's
// optimize_layout_euclidean (move_other=True, densmap off) with the same edge
// schedule (epochs_per_sample), the same per-vertex Tausworthe RNG streams and
// the same float32-coordinate / float64-gradient arithmetic.  umap-learn
// compiles this loop with numba fastmath, so results agree to rounding, not
// bit for bit.  The sequential version is deterministic; the optional
// Hogwild version (like umap-learn's parallel=True) is not.
#pragma once

#include <cstdint>

namespace aof {

struct UmapSgdParams {
    double a = 1.577, b = 0.895;
    double gamma = 1.0;
    double initial_alpha = 1.0;
    double negative_sample_rate = 5.0;
    int n_epochs = 200;
    bool move_other = true;
    bool parallel = false;
};

// embedding: n x dim float32 (row-major), updated in place.
// head/tail/epochs_per_sample: one entry per edge (graph.row, graph.col,
// make_epochs_per_sample(graph.data)).  rng_state: 3 int64 seeds
// (random_state.randint(INT32_MIN, INT32_MAX, 3)).
void umap_optimize_euclidean(float* embedding, int64_t n, int dim, const int32_t* head,
                             const int32_t* tail, int64_t n_edges, const double* epochs_per_sample,
                             const int64_t* rng_state, const UmapSgdParams& p);

// Deterministic parallel variant.  Edges are given in CSR order by head
// (indptr over vertices, tail[e], epochs_per_sample[e]) of a symmetric graph.
// Vertices are processed in fixed blocks of `block` in index order (block
// Gauss-Seidel, close to umap's sequential edge order); within a block each
// vertex is updated by one thread from its own edges only (the attractive
// move it would receive as the tail of the reverse edge, which has the same
// schedule, is applied with it), reading coordinates as of the block start.
// Negative samples come from a stateless hash of (seed, edge, epoch).  The
// result does not depend on the number of threads or on scheduling.
void umap_optimize_euclidean_det(float* embedding, int64_t n, int dim, const int64_t* indptr,
                                 const int32_t* tail, const double* epochs_per_sample, uint64_t seed,
                                 const UmapSgdParams& p, int64_t block = 0);

// y = A x for a CSR float64 matrix, row-parallel (each row summed in order,
// identical to scipy's csr_matvec and independent of the thread count).
void csr_matvec(const int64_t* indptr, const int32_t* indices, const double* data, int64_t n_rows,
                const double* x, double* y);

}  // namespace aof
