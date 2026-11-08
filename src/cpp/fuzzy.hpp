// UMAP fuzzy simplicial set (umap-learn 0.5 `fuzzy_simplicial_set` on a
// precomputed kNN graph), reproducing its float32 arithmetic step by step:
// smooth_knn_dist's binary search runs in float32 locals exactly as umap's
// numba kernel declares them, membership strengths use float32 expf, and the
// fuzzy union  (A + A^T) - A o A^T  rounds in the same order as scipy.
#pragma once

#include <cstdint>
#include <vector>

namespace aof {

struct FuzzyResult {
    std::vector<int64_t> indptr;
    std::vector<int32_t> indices;
    std::vector<float> data;
    std::vector<float> sigmas, rhos;
};

// knn_indices / knn_dists: n x k row-major, each row sorted by distance and
// normally starting with the point itself (distance 0).  Entries < 0 are
// treated as missing.
FuzzyResult fuzzy_simplicial_set(const int64_t* knn_indices, const float* knn_dists, int64_t n,
                                 int64_t k, double n_neighbors, double local_connectivity,
                                 double set_op_mix_ratio, double bandwidth);

}  // namespace aof
