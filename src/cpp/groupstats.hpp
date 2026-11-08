// Per-group, per-gene statistics of an in-memory CSR matrix (cells x genes):
// sum, sum of squares and stored-entry count for every (group, gene).
//
// Rows are visited group by group in a caller-supplied order; within a group
// they are split into fixed chunks reduced in chunk order, so the sums do not
// depend on the number of threads.  Memory is O(slots x genes), independent of
// the number of groups.
#pragma once

#include <cstdint>
#include <vector>

namespace aof {

struct GroupStats {
    std::vector<double> sum, sumsq, nnz;  // n_groups x n_cols, row-major
};

// order: row indices grouped by group (group g = order[offsets[g]:offsets[g+1]]).
template <class V, class I>
GroupStats group_stats_csr(const int64_t* indptr, const I* indices, const V* data, int64_t n_cols,
                           const int64_t* order, const int64_t* offsets, int64_t n_groups);

}  // namespace aof
