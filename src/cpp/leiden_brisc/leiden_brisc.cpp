// C++ port of brisc's Leiden clustering (brisc/leiden.pyx `leiden` /
// `leiden_nogil`; RNG from brisc/cyutils.pxd `srand` / `rand` / `randint`).
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
// The algorithm (a sequential, deterministic GVE-Leiden variant optimising
// the RBConfiguration / modularity objective), the PCG32 RNG, the visiting
// order, every floating point expression (float32 edge weights, double
// accumulators, same operation order, no FMA contraction) and the label
// numbering are reproduced exactly, so the labels are identical to brisc's
// element for element.  The buffer sharing and software prefetching of the
// original are kept.  Differences: no KeyboardInterrupt polling, and the
// one-element over-read that brisc's prefetch of
// `communities[indices[indptr[i]]]` can make for an empty last row is
// guarded.
//
// This port is part of anndataoom-fast, GNU GPL-3.0-or-later.

#include "leiden_brisc.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#define AOF_BRISC_PREFETCH(addr) __builtin_prefetch(addr)
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <xmmintrin.h>
#define AOF_BRISC_PREFETCH(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#else
#define AOF_BRISC_PREFETCH(addr) ((void)0)
#endif

namespace aof_brisc {

namespace {

// --- brisc/cyutils.pxd ------------------------------------------------------

inline uint32_t pcg_rand(uint64_t* state) {
    const uint64_t x = *state;
    *state = x * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    const uint32_t s = uint32_t((x ^ (x >> 18)) >> 27);
    const uint32_t rot = uint32_t(x >> 59);
    return (s >> rot) | (s << ((0u - rot) & 31u));
}

// Lemire's method
inline uint32_t pcg_randint(const uint32_t bound, uint64_t* state) {
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

inline uint64_t pcg_srand(const uint64_t seed) {
    uint64_t state = seed + UINT64_C(1442695040888963407);
    pcg_rand(&state);
    return state;
}

// --- brisc/leiden.pyx -------------------------------------------------------

inline uint32_t next_pow2_minus_1(uint32_t x) {
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x;
}

template <typename T>
std::unique_ptr<T[]> uninitialized(uint64_t n) {
    return std::unique_ptr<T[]>(new T[n > 0 ? size_t(n) : 1]);
}

// Python's repr() of a float32 promoted to double (shortest round-trip form).
std::string py_float_repr(float r) {
    const double d = double(r);
    char buf[64];
    for (int prec = 1; prec <= 17; prec++) {
        std::snprintf(buf, sizeof(buf), "%.*g", prec, d);
        if (std::strtod(buf, nullptr) == d) break;
    }
    std::string s(buf);
    if (s.find_first_of(".eEni") == std::string::npos) s += ".0";
    return s;
}

std::string with_commas(uint64_t v) {
    std::string s = std::to_string(v);
    for (int p = int(s.size()) - 3; p > 0; p -= 3) s.insert(size_t(p), ",");
    return s;
}

}  // namespace

template <typename Idx>
uint32_t leiden(const float* data_original, const Idx* indices_original,
                const Idx* indptr_original, const uint64_t num_cells, const Neighbors& neighbors,
                uint32_t* final_communities, const float resolution,
                const uint32_t min_cluster_size, const uint64_t seed,
                const bool resolution_in_message) {
    // Note: this code extensively exploits the fact that the number of nodes,
    // edges, communities etc. gets smaller on each iteration, so everything
    // is preallocated once and only a fraction is used later on.  The input
    // graph must not have self-edges; aggregated graphs do, and each node's
    // self-edge is written first in its row.

    uint32_t i, k, leiden_iteration, queue_head, queue_tail, community, other, other_community,
        new_community = 0, num_refined_communities, num_final_communities, community_size,
        neighbor, future_i, unrefined_community, contiguous_index, original_index, current_sum,
        num_touched;
    uint64_t num_nodes, word_index, bit_index, num_edges, edges_written;
    const uint64_t queue_size_minus_1 = next_pow2_minus_1(uint32_t(num_cells));
    const uint64_t queue_size = queue_size_minus_1 + 1;
    uint64_t state = pcg_srand(seed);
    Idx start_index, end_index, j;
    double weighted_degree, total_weighted_degree, scaled_resolution, community_weight,
        best_delta_objective, scaled_resolution_times_weighted_degree,
        other_node_to_community_weight, other_community_weight, delta_objective, edge_weight;
    bool any_moved;

    auto communities_buffer = uninitialized<uint32_t>(num_cells);
    auto node_order_buffer = uninitialized<uint32_t>(num_cells);
    auto queue_buffer = uninitialized<uint32_t>(queue_size);
    auto weighted_degrees_buffer = uninitialized<double>(num_cells);
    auto community_weights_buffer = uninitialized<double>(num_cells);
    auto refined_community_weights_buffer = uninitialized<double>(num_cells);
    std::vector<double> node_to_community_weights_buffer(size_t(num_cells), 0.0);
    std::vector<uint64_t> bitset_buffer(size_t((num_cells + 63) / 64), 0);
    std::unique_ptr<uint32_t[]> refined_communities_buffer, community_node_indptr_buffer;
    std::unique_ptr<float[]> data_buffer, data_buffer_alt;
    std::unique_ptr<Idx[]> indices_buffer, indices_buffer_alt, indptr_buffer, indptr_buffer_alt;

    // Memory shared between mutually exclusive phases, exactly as in brisc:
    // `node_order` and `queue` are only used in the move phase,
    // `original_to_contiguous` only in the refinement phase and at the very
    // end, `contiguous_to_original` only in the refinement phase,
    // `community_nodes` and `touched_communities` only in the aggregation
    // phase, and `community_sizes` only at the very end.  On the first
    // iteration `refined_communities` is the output array itself.
    uint32_t* contiguous_community_sizes = nullptr;
    uint32_t* community_node_indptr = nullptr;
    uint32_t* refined_communities = final_communities;
    uint32_t* const communities = communities_buffer.get();
    uint32_t* const node_order = node_order_buffer.get();
    uint32_t* const queue = queue_buffer.get();
    uint32_t* const original_to_contiguous = queue;
    uint32_t* const contiguous_to_original = node_order;
    uint32_t* const community_nodes = node_order;
    uint32_t* const touched_communities = queue;
    uint32_t* const community_sizes = node_order;
    double* const weighted_degrees = weighted_degrees_buffer.get();
    double* const community_weights = community_weights_buffer.get();
    double* const refined_community_weights = refined_community_weights_buffer.get();
    double* const node_to_community_weights = node_to_community_weights_buffer.data();
    uint64_t* const bitset = bitset_buffer.data();

    const float* data = data_original;
    const Idx* indices = indices_original;
    const Idx* indptr = indptr_original;
    float* next_data;
    Idx* next_indices;
    Idx* next_indptr;

    // Weighted degrees, 2m, and the singleton partition
    total_weighted_degree = 0;
    for (uint64_t c = 0; c < num_cells; c++) {
        weighted_degree = 0;
        start_index = indptr[c];
        end_index = indptr[c + 1];
        for (j = start_index; j < end_index; j++) {
            edge_weight = data[j];
            weighted_degree += edge_weight;
        }
        weighted_degrees[c] = weighted_degree;
        total_weighted_degree += weighted_degree;
        communities[c] = uint32_t(c);
        community_weights[c] = weighted_degree;
    }

    scaled_resolution = double(resolution) / total_weighted_degree;

    // Evaluate moving node `i` to the best neighbouring community (move
    // phase), shared by the initial pass and the queue pass.  Returns true if
    // the node moved.
    auto move_node = [&](const uint32_t node) -> bool {
        const uint32_t i_ = node;
        const uint32_t community_ = communities[i_];
        const double weighted_degree_ = weighted_degrees[i_];
        const double community_weight_ = community_weights[community_];

        Idx s = indptr[i_];
        const Idx e = indptr[i_ + 1];
        if (leiden_iteration && s != e && uint32_t(indices[s]) == i_) s += 1;
        Idx jj = s;
        while (jj < e - 16) {
            AOF_BRISC_PREFETCH(&communities[indices[jj + 16]]);
            const uint32_t oc = communities[indices[jj]];
            node_to_community_weights[oc] += double(data[jj]);
            AOF_BRISC_PREFETCH(&community_weights[oc]);
            jj += 1;
        }
        while (jj < e) {
            const uint32_t oc = communities[indices[jj]];
            node_to_community_weights[oc] += double(data[jj]);
            AOF_BRISC_PREFETCH(&community_weights[oc]);
            jj += 1;
        }
        const double node_to_community_weight_ = node_to_community_weights[community_];
        node_to_community_weights[community_] = 0;

        const double srtwd = scaled_resolution * weighted_degree_;
        const double base = node_to_community_weight_ +
                            srtwd * (weighted_degree_ - community_weight_);
        double best = base;
        uint32_t new_comm = new_community;
        for (jj = s; jj < e; jj++) {
            const uint32_t oc = communities[indices[jj]];
            const double ontcw = node_to_community_weights[oc];
            if (ontcw > 0) {
                node_to_community_weights[oc] = 0;
                const double ocw = community_weights[oc];
                const double delta = ontcw - srtwd * ocw;
                if (delta > best) {
                    best = delta;
                    new_comm = oc;
                }
            }
        }
        new_community = new_comm;

        if (best > base) {
            communities[i_] = new_comm;
            community_weights[new_comm] += weighted_degree_;
            community_weights[community_] -= weighted_degree_;
            for (jj = s; jj < e; jj++) {
                const uint32_t o = uint32_t(indices[jj]);
                if (communities[o] != new_comm) {
                    const uint64_t wi = uint64_t(o) >> 6;
                    const uint64_t bi = uint64_t(o) & 63;
                    if (!(bitset[wi] & (UINT64_C(1) << bi))) {
                        bitset[wi] |= UINT64_C(1) << bi;
                        queue[queue_tail & queue_size_minus_1] = o;
                        queue_tail += 1;
                    }
                }
            }
            return true;
        }
        return false;
    };

    leiden_iteration = 0;
    num_nodes = num_cells;
    while (true) {
        // Number of edges of the current graph, to keep the prefetch of
        // `indices[indptr[x]]` in bounds when row x is empty.
        const Idx nnz = indptr[num_nodes];

        // 1) Move phase.  Random visiting order via the "inside-out"
        //    Fisher-Yates shuffle (brisc reads the uninitialised slot when
        //    i == k and overwrites it immediately; skip that self-copy).
        for (k = 0; k < num_nodes; k++) {
            i = pcg_randint(k + 1, &state);
            if (i != k) node_order[k] = node_order[i];
            node_order[i] = k;
        }

        queue_head = queue_tail = 0;
        any_moved = false;
        for (k = 0; k < num_nodes; k++) {
            if (k + 24 < num_nodes) AOF_BRISC_PREFETCH(&indptr[node_order[k + 24]]);
            if (k + 16 < num_nodes) {
                const uint32_t f = node_order[k + 16];
                AOF_BRISC_PREFETCH(&indices[indptr[f]]);
                AOF_BRISC_PREFETCH(&data[indptr[f]]);
                AOF_BRISC_PREFETCH(&communities[f]);
            }
            if (k + 8 < num_nodes) {
                const uint32_t f = node_order[k + 8];
                const Idx p = indptr[f];
                if (p < nnz) AOF_BRISC_PREFETCH(&communities[indices[p]]);
                AOF_BRISC_PREFETCH(&weighted_degrees[f]);
                AOF_BRISC_PREFETCH(&community_weights[communities[f]]);
            }
            if (move_node(node_order[k])) any_moved = true;
        }

        // Globally converged: no node moved in this iteration's move phase
        if (!any_moved) {
            if (leiden_iteration == 0) {
                std::string msg = "every cell ended up in a cluster by itself";
                if (resolution > 1.0f) msg += "; consider decreasing resolution";
                if (resolution_in_message)
                    msg += " (resolution = " + py_float_repr(resolution) + ")";
                throw std::invalid_argument(msg);
            }

            // Each cell's refined community -> its final (move-phase) cluster
            for (uint64_t c = 0; c < num_cells; c++)
                final_communities[c] = communities[final_communities[c]];

            // Merge cells in clusters smaller than `min_cluster_size` into the
            // cluster of their nearest neighbour that is in a cluster of size
            // >= `min_cluster_size`; relabel contiguously, skipping the
            // too-small clusters.
            std::fill(community_sizes, community_sizes + num_cells, 0u);
            for (uint64_t c = 0; c < num_cells; c++) community_sizes[final_communities[c]] += 1;
            num_final_communities = 0;
            for (uint64_t c = 0; c < num_cells; c++) {
                community_size = community_sizes[c];
                if (community_size >= min_cluster_size) {
                    original_to_contiguous[c] = num_final_communities;
                    contiguous_community_sizes[num_final_communities] = community_size;
                    num_final_communities += 1;
                }
            }
            for (uint64_t c = 0; c < num_cells; c++) {
                i = uint32_t(c);
                community = final_communities[i];
                if (community_sizes[community] < min_cluster_size) {
                    const uint32_t* row;
                    uint64_t row_len;
                    if (neighbors.indptr) {
                        row = neighbors.nbr + neighbors.indptr[c];
                        row_len = uint64_t(neighbors.indptr[c + 1] - neighbors.indptr[c]);
                    } else {
                        row = neighbors.nbr + c * neighbors.k;
                        row_len = neighbors.k;
                    }
                    bool assigned = false;
                    for (uint64_t t = 0; t < row_len; t++) {
                        neighbor = row[t];
                        if (neighbor >= num_cells)
                            throw std::invalid_argument(
                                "some nearest-neighbor indices are >= the total number of "
                                "cells, " + with_commas(num_cells) + ".");
                        if (neighbor == i)
                            throw std::invalid_argument(
                                "some nearest-neighbor indices indicate that a cell is its "
                                "own neighbor, i.e. neighbors[i, j] == i for some i and j.");
                        other_community = final_communities[neighbor];
                        if (neighbor < i) {
                            // already mapped to a contiguous label
                            if (contiguous_community_sizes[other_community] >= min_cluster_size) {
                                final_communities[i] = other_community;
                                assigned = true;
                                break;
                            }
                        } else {
                            if (community_sizes[other_community] >= min_cluster_size) {
                                final_communities[i] = original_to_contiguous[other_community];
                                assigned = true;
                                break;
                            }
                        }
                    }
                    if (!assigned) {
                        std::string msg = "cell " + std::to_string(i) + " and all of its " +
                                          with_commas(row_len) +
                                          " nearest neighbors are in clusters smaller than "
                                          "min_cluster_size (" +
                                          with_commas(min_cluster_size) +
                                          "), so it cannot be assigned a cluster label; "
                                          "consider decreasing min_cluster_size";
                        if (resolution > 1.0f) msg += " and/or resolution";
                        if (resolution_in_message)
                            msg += " (resolution = " + py_float_repr(resolution) + ")";
                        throw std::invalid_argument(msg);
                    }
                } else {
                    final_communities[i] = original_to_contiguous[community];
                }
            }
            return num_final_communities;
        }

        // Keep moving nodes until the queue is empty
        while (queue_head != queue_tail) {
            if (queue_head + 24 < queue_tail)
                AOF_BRISC_PREFETCH(&indptr[queue[(queue_head + 24) & queue_size_minus_1]]);
            if (queue_head + 16 < queue_tail) {
                const uint32_t f = queue[(queue_head + 16) & queue_size_minus_1];
                AOF_BRISC_PREFETCH(&indices[indptr[f]]);
                AOF_BRISC_PREFETCH(&data[indptr[f]]);
                AOF_BRISC_PREFETCH(&communities[f]);
            }
            if (queue_head + 8 < queue_tail) {
                future_i = queue[(queue_head + 8) & queue_size_minus_1];
                const Idx p = indptr[future_i];
                if (p < nnz) AOF_BRISC_PREFETCH(&communities[indices[p]]);
                AOF_BRISC_PREFETCH(&weighted_degrees[future_i]);
                AOF_BRISC_PREFETCH(&community_weights[communities[future_i]]);
            }

            i = queue[queue_head & queue_size_minus_1];
            word_index = uint64_t(i) >> 6;
            bit_index = uint64_t(i) & 63;
            bitset[word_index] &= ~(UINT64_C(1) << bit_index);
            queue_head += 1;
            move_node(i);
        }

        // 2) Refinement phase: every node starts in its own refined community;
        //    a single pass merges singleton nodes into refined communities
        //    within their move-phase community.
        for (k = 0; k < num_nodes; k++) {
            refined_communities[k] = k;
            refined_community_weights[k] = weighted_degrees[k];
        }

        num_refined_communities = 0;
        for (i = 0; i < num_nodes; i++) {
            community = refined_communities[i];
            weighted_degree = weighted_degrees[i];
            community_weight = refined_community_weights[community];
            if (weighted_degree != community_weight) {
                original_to_contiguous[community] = num_refined_communities;
                contiguous_to_original[num_refined_communities] = community;
                num_refined_communities += 1;
                continue;
            }

            unrefined_community = communities[i];
            start_index = indptr[i];
            end_index = indptr[i + 1];
            if (leiden_iteration && start_index != end_index &&
                uint32_t(indices[start_index]) == i)
                start_index += 1;
            j = start_index;
            while (j < end_index - 16) {
                AOF_BRISC_PREFETCH(&refined_communities[indices[j + 16]]);
                other = uint32_t(indices[j]);
                other_community = refined_communities[other];
                node_to_community_weights[other_community] += double(data[j]);
                AOF_BRISC_PREFETCH(&communities[other]);
                AOF_BRISC_PREFETCH(&refined_community_weights[other_community]);
                j += 1;
            }
            while (j < end_index) {
                other = uint32_t(indices[j]);
                other_community = refined_communities[other];
                node_to_community_weights[other_community] += double(data[j]);
                AOF_BRISC_PREFETCH(&communities[other]);
                AOF_BRISC_PREFETCH(&refined_community_weights[other_community]);
                j += 1;
            }

            scaled_resolution_times_weighted_degree = scaled_resolution * weighted_degree;
            best_delta_objective = 0;
            for (j = start_index; j < end_index; j++) {
                other = uint32_t(indices[j]);
                other_community = refined_communities[other];
                other_node_to_community_weight = node_to_community_weights[other_community];
                if (other_node_to_community_weight > 0) {
                    node_to_community_weights[other_community] = 0;
                    if (communities[other] == unrefined_community) {
                        other_community_weight = refined_community_weights[other_community];
                        delta_objective = other_node_to_community_weight -
                                          scaled_resolution_times_weighted_degree *
                                              other_community_weight;
                        if (delta_objective > best_delta_objective) {
                            best_delta_objective = delta_objective;
                            new_community = other_community;
                        }
                    }
                }
            }

            if (best_delta_objective > 0) {
                refined_communities[i] = new_community;
                refined_community_weights[new_community] += weighted_degree;
            } else {
                original_to_contiguous[community] = num_refined_communities;
                contiguous_to_original[num_refined_communities] = community;
                num_refined_communities += 1;
            }
        }

        if (leiden_iteration == 0) {
            community_node_indptr_buffer = uninitialized<uint32_t>(uint64_t(num_refined_communities) + 1);
            community_node_indptr = community_node_indptr_buffer.get();
        }
        for (k = 0; k < num_refined_communities; k++) community_node_indptr[k] = 0;

        // Relabel refined communities contiguously and count their sizes
        for (k = 0; k < num_nodes; k++) {
            community = original_to_contiguous[refined_communities[k]];
            refined_communities[k] = community;
            community_node_indptr[community] += 1;
        }

        // Supergraph nodes inherit the unrefined community of their refined
        // community, and the refined community weight as weighted degree
        for (contiguous_index = 0; contiguous_index < num_refined_communities;
             contiguous_index++) {
            if (contiguous_index + 8 < num_refined_communities) {
                future_i = contiguous_to_original[contiguous_index + 8];
                AOF_BRISC_PREFETCH(&communities[future_i]);
                AOF_BRISC_PREFETCH(&refined_community_weights[future_i]);
            }
            original_index = contiguous_to_original[contiguous_index];
            communities[contiguous_index] = communities[original_index];
            weighted_degrees[contiguous_index] = refined_community_weights[original_index];
        }

        // 3) Aggregation phase
        if (leiden_iteration > 0) {
            for (uint64_t c = 0; c < num_cells; c++)
                final_communities[c] = refined_communities[final_communities[c]];
        }

        // Counting sort of nodes by refined community
        current_sum = 0;
        for (k = 0; k < num_refined_communities; k++) {
            current_sum += community_node_indptr[k];
            community_node_indptr[k] = current_sum;
        }
        community_node_indptr[num_refined_communities] = current_sum;

        i = uint32_t(num_nodes);
        while (i > 0) {
            i -= 1;
            community = refined_communities[i];
            community_node_indptr[community] -= 1;
            community_nodes[community_node_indptr[community]] = i;
        }

        // Ping-pong buffers for the supergraph, allocated on the first two
        // iterations (at most min(R^2, current nnz) edges)
        if (leiden_iteration % 2 == 0) {
            if (leiden_iteration == 0) {
                num_edges = std::min(uint64_t(num_refined_communities) * num_refined_communities,
                                     uint64_t(indptr[num_nodes]));
                data_buffer = uninitialized<float>(num_edges);
                indices_buffer = uninitialized<Idx>(num_edges);
                indptr_buffer = uninitialized<Idx>(uint64_t(num_refined_communities) + 1);
            }
            next_data = data_buffer.get();
            next_indices = indices_buffer.get();
            next_indptr = indptr_buffer.get();
        } else {
            if (leiden_iteration == 1) {
                num_edges = std::min(uint64_t(num_refined_communities) * num_refined_communities,
                                     uint64_t(indptr[num_nodes]));
                data_buffer_alt = uninitialized<float>(num_edges);
                indices_buffer_alt = uninitialized<Idx>(num_edges);
                indptr_buffer_alt = uninitialized<Idx>(uint64_t(num_refined_communities) + 1);
            }
            next_data = data_buffer_alt.get();
            next_indices = indices_buffer_alt.get();
            next_indptr = indptr_buffer_alt.get();
        }

        edges_written = 0;
        next_indptr[0] = 0;
        for (community = 0; community < num_refined_communities; community++) {
            num_touched = 0;
            for (k = community_node_indptr[community]; k < community_node_indptr[community + 1];
                 k++) {
                if (k + 8 < num_nodes) AOF_BRISC_PREFETCH(&indptr[community_nodes[k + 8]]);
                if (k + 4 < num_nodes) {
                    const uint32_t f = community_nodes[k + 4];
                    AOF_BRISC_PREFETCH(&indices[indptr[f]]);
                    AOF_BRISC_PREFETCH(&data[indptr[f]]);
                }
                i = community_nodes[k];
                start_index = indptr[i];
                end_index = indptr[i + 1];
                // Self-loop (first in its row): added twice to cancel the
                // halving below.
                if (leiden_iteration && start_index != end_index &&
                    uint32_t(indices[start_index]) == i) {
                    if (node_to_community_weights[community] == 0) {
                        touched_communities[num_touched] = community;
                        num_touched += 1;
                    }
                    node_to_community_weights[community] += 2 * double(data[start_index]);
                    start_index += 1;
                }
                j = start_index;
                while (j < end_index - 16) {
                    AOF_BRISC_PREFETCH(&refined_communities[indices[j + 16]]);
                    other = uint32_t(indices[j]);
                    other_community = refined_communities[other];
                    edge_weight = data[j];
                    if (node_to_community_weights[other_community] == 0) {
                        touched_communities[num_touched] = other_community;
                        num_touched += 1;
                    }
                    node_to_community_weights[other_community] += edge_weight;
                    j += 1;
                }
                while (j < end_index) {
                    other = uint32_t(indices[j]);
                    other_community = refined_communities[other];
                    edge_weight = data[j];
                    if (node_to_community_weights[other_community] == 0) {
                        touched_communities[num_touched] = other_community;
                        num_touched += 1;
                    }
                    node_to_community_weights[other_community] += edge_weight;
                    j += 1;
                }
            }

            // Halve intra-community edges (symmetric double counting)
            node_to_community_weights[community] *= 0.5;

            // Self-edge first, then the remaining edges in touch order
            if (node_to_community_weights[community] != 0) {
                next_data[edges_written] = float(node_to_community_weights[community]);
                next_indices[edges_written] = Idx(community);
                edges_written += 1;
                node_to_community_weights[community] = 0;
            }
            for (k = 0; k < num_touched; k++) {
                other_community = touched_communities[k];
                if (other_community != community) {
                    next_data[edges_written] = float(node_to_community_weights[other_community]);
                    next_indices[edges_written] = Idx(other_community);
                    edges_written += 1;
                    node_to_community_weights[other_community] = 0;
                }
            }
            next_indptr[community + 1] = Idx(edges_written);
        }

        if (leiden_iteration == 0) {
            refined_communities_buffer = uninitialized<uint32_t>(num_cells);
            refined_communities = refined_communities_buffer.get();
            contiguous_community_sizes = refined_communities;
        }

        data = next_data;
        indices = next_indices;
        indptr = next_indptr;
        num_nodes = num_refined_communities;
        leiden_iteration += 1;
    }
}

template uint32_t leiden<int32_t>(const float*, const int32_t*, const int32_t*, uint64_t,
                                  const Neighbors&, uint32_t*, float, uint32_t, uint64_t, bool);
template uint32_t leiden<int64_t>(const float*, const int64_t*, const int64_t*, uint64_t,
                                  const Neighbors&, uint32_t*, float, uint32_t, uint64_t, bool);

}  // namespace aof_brisc
