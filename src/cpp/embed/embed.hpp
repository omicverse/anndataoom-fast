// C++ port of brisc's PaCMAP and LocalMAP embeddings (brisc/embed.pyx:
// `pacmap`, `localmap` and their helpers; RNG from brisc/cyutils.pxd).
//
// brisc: Copyright (c) 2026 Michael Wainberg and Keon Arbabi, MIT License
// (https://github.com/briscverse/brisc).  The MIT notice is reproduced in
// embed.cpp and in the package LICENSE.
//
// This port is part of anndataoom-fast and is distributed under the GNU
// General Public License, version 3 or later.
#pragma once

#include <cstddef>
#include <cstdint>

namespace aof_embed {

struct Params {
    uint32_t num_neighbors = 10;       // nearest-neighbour pairs kept per cell
    uint32_t num_mid_near_pairs = 5;
    uint32_t num_further_pairs = 20;
    uint32_t num_phase_1_iterations = 100;
    uint32_t num_phase_2_iterations = 100;
    uint32_t num_phase_3_iterations = 250;
    float learning_rate = 1.0f;
    float max_distance = 10.0f;        // LocalMAP only
    uint64_t seed = 0;
    bool localmap = false;             // false: PaCMAP, true: LocalMAP
    int num_threads = 0;               // <= 0: omp_get_max_threads()
};

// Optional outputs for testing: the sampled pair lists (n x num_*), each may be
// null.  `further_pairs` receives the final further pairs (after LocalMAP's
// re-sampling).
struct PairsOut {
    uint32_t* neighbor_pairs = nullptr;
    uint32_t* mid_near_pairs = nullptr;
    uint32_t* further_pairs = nullptr;
};

// Embed `n` cells with `d`-dimensional coordinates `PCs` (row-major float32)
// into `embedding` (n x 2, row-major float32).  `knn` / `knn_sq_dist` are the
// n x k nearest neighbours of every cell (excluding the cell itself), nearest
// first, with squared Euclidean distances -- brisc's obsm['neighbors'] and
// obsm['distances'].  All k columns are used, as in brisc.
//
// Element-for-element identical to brisc's parallel code path (the result for
// any `num_threads > 1`, or `num_threads=1, match_parallel=True`), on x86 CPUs;
// the output does not depend on the number of threads.  Throws
// std::invalid_argument on invalid input.
void embed(const float* PCs, uint64_t n, uint64_t d, const uint32_t* knn,
           const float* knn_sq_dist, uint64_t k, const Params& params,
           float* embedding, const PairsOut& pairs_out = PairsOut());

}  // namespace aof_embed
