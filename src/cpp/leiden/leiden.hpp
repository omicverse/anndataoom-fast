// Exact C++ port of leidenalg.find_partition(..., RBConfigurationVertexPartition,
// n_iterations=..., seed=...) as shipped in leidenalg 0.12.0 / libleidenalg 0.13.0
// on top of the igraph C core 1.0.
//
// Ported from:
//   * libleidenalg (GraphHelper.cpp, MutableVertexPartition.cpp,
//     RBConfigurationVertexPartition.cpp, Optimiser.cpp) and the leidenalg
//     Python glue (functions.py, Optimiser.py, python_*_interface.cpp),
//     Copyright (C) V.A. Traag, GNU GPL-3.0-or-later.
//   * igraph C core (src/random/random.c, src/random/rng_mt19937.c,
//     src/core/vector.c igraph_vector_int_pair_order,
//     src/graph/type_indexededgelist.c igraph_neighbors / igraph_incident /
//     igraph_add_edges / igraph_get_eid), Copyright (C) The igraph development
//     team, GNU GPL-2.0-or-later.
//
// This file is part of anndataoom-fast and is distributed under the GNU
// General Public License, version 3 or later.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aof_leiden {

// Immutable graph with igraph's storage semantics (edge ids = insertion
// order, undirected edges stored as (max, min)) and leidenalg's Graph
// bookkeeping (strengths, self weights, node sizes, total weight).
//
// Only the IGRAPH_ALL neighbour list is materialised, in exactly the order
// igraph_neighbors()/igraph_incident() return it (IGRAPH_LOOPS_TWICE,
// IGRAPH_MULTIPLE).  For directed graphs it is the merge of the OUT and IN
// lists, and each entry carries a flag telling which of the two it came from;
// a merge preserves the relative order of each input, so the OUT (resp. IN)
// list is exactly the subsequence of OUT- (resp. IN-) flagged entries.
// For undirected graphs igraph ignores the mode: OUT == IN == ALL.
struct Graph {
    static constexpr uint32_t IN_FLAG = UINT32_C(0x80000000);
    static constexpr uint32_t ID_MASK = UINT32_C(0x7fffffff);

    size_t n = 0, m = 0;
    bool directed = true;
    bool all_positive = true;  // every edge weight > 0

    std::vector<uint32_t> efrom, eto;  // stored endpoints, edge id order
    std::vector<double> ew;            // edge weights
    std::vector<double> node_size, self_weight, strength_in, strength_out;
    double total_weight = 0.0, total_size = 0.0;

    std::vector<size_t> off;      // ALL list offsets (n + 1)
    std::vector<uint32_t> nb;     // neighbour id | IN_FLAG (directed, from in-list)
    std::vector<double> w;        // edge weight of the entry
    std::vector<uint32_t> outcnt; // undirected: # entries of the "out" part per vertex
    // directed only: OUT list (igraph_incident(mode=OUT) order) for collapse_graph
    std::vector<size_t> out_off;
    std::vector<uint32_t> out_nb;
    std::vector<double> out_w;

    inline double strength_out_(size_t v) const { return directed ? strength_out[v] : strength_in[v]; }

    // edges: pairs (a_k, b_k) as passed to igraph_add_edges / igraph_create.
    // node_sizes may be empty (=> all 1.0).
    static Graph* create(size_t n, bool directed, const std::vector<uint32_t>& a,
                         const std::vector<uint32_t>& b, std::vector<double>&& weights,
                         std::vector<double>&& node_sizes);

    // Directed graph whose edges are grouped by source in increasing order,
    // with distinct targets per source (what collapse_graph() produces).
    // Builds exactly the same structure as create() would; the work is split
    // over `nthreads` OpenMP threads without changing any summation order.
    static Graph* create_grouped_directed(size_t n, std::vector<uint32_t>&& from,
                                          std::vector<uint32_t>&& to, std::vector<double>&& weights,
                                          std::vector<double>&& node_sizes, int nthreads);
};

struct Options {
    int n_iterations = -1;  // < 0: iterate until an iteration brings no improvement
    int n_threads = 1;      // threads for graph aggregation (results do not depend on it)
};

// Runs the full leidenalg.find_partition pipeline on `g` (shared, read-only)
// and returns partition.membership.
std::vector<int64_t> find_partition(const Graph& g, double resolution, uint64_t seed,
                                    const Options& opt, double* quality_gain = nullptr);

}  // namespace aof_leiden
