// Approximate (or exact) k-nearest-neighbour search in a low-dimensional
// embedding (typically PCs), after brisc's blocked inverted-file index.
//
//   1. k-means into C clusters (Lloyd iterations, fixed-block reductions).
//   2. Centroids are renumbered along their first coordinate and cells are
//      sorted by cluster, so a cluster is a contiguous column range of the
//      transposed matrix and neighbouring clusters sit close in memory.
//   3. Queries are processed in blocks of consecutive (sorted) cells; each
//      cluster probed by any query of the block is streamed once for all
//      queries that need it.  Distances use ||c||^2 - 2 q.c (the ||q||^2 term
//      does not change the ranking), and groups of 64 candidates whose best
//      bound cannot beat the current k-th distance are skipped.
//   4. The k winners are re-ranked by exactly recomputed distances.
//
// Every query is computed by one thread from read-only data, so the result is
// independent of the number of threads.  probe = n_clusters gives exact
// brute-force neighbours.
#pragma once

#include <cstdint>
#include <vector>

namespace aof {

struct KnnParams {
    int64_t k = 15;            // neighbours per cell, excluding the cell itself
    int64_t n_clusters = 0;    // 0 -> ceil(min(4 sqrt(n), n / 100))
    int64_t n_probe = 64;      // clusters searched per query
    int kmeans_iters = 4;
    int64_t block = 256;       // queries per search block
};

struct KnnResult {
    std::vector<int64_t> indices;   // n x k, excluding self, nearest first
    std::vector<float> distances;   // n x k, Euclidean (not squared)
    int64_t n_clusters = 0, n_probe = 0;
};

// X: n x d row-major float32.  init: n_clusters distinct row indices used as
// the initial centroids (chosen by the caller's RNG).
KnnResult knn_ivf(const float* X, int64_t n, int64_t d, const KnnParams& p,
                  const std::vector<int64_t>& init);

}  // namespace aof
