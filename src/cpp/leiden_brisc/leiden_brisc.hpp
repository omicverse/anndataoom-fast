// C++ port of brisc's Leiden clustering (brisc/leiden.pyx: `leiden`,
// `leiden_nogil`, `leiden_multiresolution`, plus the PCG RNG from
// brisc/cyutils.pxd).
//
// brisc: Copyright (c) 2026 Michael Wainberg and Keon Arbabi, MIT License
// (https://github.com/briscverse/brisc).  The MIT notice is reproduced in
// leiden_brisc.cpp and in the package LICENSE.
//
// This port is part of anndataoom-fast and is distributed under the GNU
// General Public License, version 3 or later.
#pragma once

#include <cstddef>
#include <cstdint>

namespace aof_brisc {

// Optional nearest-neighbour lists used to merge clusters smaller than
// `min_cluster_size` (brisc's obsm['neighbors']).  Row i is
// nbr[i*k : (i+1)*k] when `indptr` is null (a dense C-contiguous n x k uint32
// array, exactly brisc's layout) or nbr[indptr[i] : indptr[i+1]] otherwise.
struct Neighbors {
    const uint32_t* nbr = nullptr;
    const int64_t* indptr = nullptr;
    uint64_t k = 0;
};

// One Leiden run, element-for-element identical to brisc.leiden.leiden() /
// leiden_nogil().  `data`/`indices`/`indptr` is a symmetric CSR graph with
// float32 weights, no self loops and no explicit zeros.  Writes the final
// labels (0 .. n_clusters-1, brisc's numbering) to `labels` and returns the
// number of clusters.  Throws std::invalid_argument with brisc's error
// messages (every cell alone / cannot assign / bad neighbour indices).
// `resolution_in_message` adds brisc's " (resolution = r)" suffix, as in its
// multi-resolution path.
template <typename Idx>
uint32_t leiden(const float* data, const Idx* indices, const Idx* indptr, uint64_t num_cells,
                const Neighbors& neighbors, uint32_t* labels, float resolution,
                uint32_t min_cluster_size, uint64_t seed, bool resolution_in_message);

extern template uint32_t leiden<int32_t>(const float*, const int32_t*, const int32_t*, uint64_t,
                                         const Neighbors&, uint32_t*, float, uint32_t, uint64_t,
                                         bool);
extern template uint32_t leiden<int64_t>(const float*, const int64_t*, const int64_t*, uint64_t,
                                         const Neighbors&, uint32_t*, float, uint32_t, uint64_t,
                                         bool);

}  // namespace aof_brisc
