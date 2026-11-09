// nanobind bindings for the brisc PaCMAP / LocalMAP port (module
// anndataoom_fast._embed_cpp).
//
// The algorithms are ported from brisc (brisc/embed.pyx), Copyright (c) 2026
// Michael Wainberg and Keon Arbabi, MIT License; see embed.cpp.  This file is
// part of anndataoom-fast, GNU GPL-3.0-or-later.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include "embed.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using F2In = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using U2In = nb::ndarray<const uint32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using F2Out = nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using U2Out = nb::ndarray<uint32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

void run(F2In pcs, U2In knn, F2In knn_sq_dist, F2Out out, bool localmap, uint32_t num_neighbors,
         uint32_t num_mid_near_pairs, uint32_t num_further_pairs, uint32_t it1, uint32_t it2,
         uint32_t it3, float learning_rate, float max_distance, uint64_t seed, int n_threads,
         std::optional<U2Out> neighbor_pairs, std::optional<U2Out> mid_near_pairs,
         std::optional<U2Out> further_pairs) {
    const uint64_t n = pcs.shape(0), d = pcs.shape(1), k = knn.shape(1);
    if (knn.shape(0) != n || knn_sq_dist.shape(0) != n || knn_sq_dist.shape(1) != k)
        throw std::invalid_argument("knn and knn_sq_dist must both have shape (n_cells, k)");
    if (out.shape(0) != n || out.shape(1) != 2)
        throw std::invalid_argument("out must have shape (n_cells, 2)");
    aof_embed::PairsOut po;
    auto check = [&](std::optional<U2Out>& a, uint32_t w, const char* name) -> uint32_t* {
        if (!a) return nullptr;
        if (a->shape(0) != n || a->shape(1) != w)
            throw std::invalid_argument(std::string(name) + " has the wrong shape");
        return a->data();
    };
    po.neighbor_pairs = check(neighbor_pairs, num_neighbors, "neighbor_pairs");
    po.mid_near_pairs = check(mid_near_pairs, num_mid_near_pairs, "mid_near_pairs");
    po.further_pairs = check(further_pairs, num_further_pairs, "further_pairs");

    aof_embed::Params p;
    p.localmap = localmap;
    p.num_neighbors = num_neighbors;
    p.num_mid_near_pairs = num_mid_near_pairs;
    p.num_further_pairs = num_further_pairs;
    p.num_phase_1_iterations = it1;
    p.num_phase_2_iterations = it2;
    p.num_phase_3_iterations = it3;
    p.learning_rate = learning_rate;
    p.max_distance = max_distance;
    p.seed = seed;
    p.num_threads = n_threads;
    nb::gil_scoped_release g;
    aof_embed::embed(pcs.data(), n, d, knn.data(), knn_sq_dist.data(), k, p, out.data(), po);
}

}  // namespace

NB_MODULE(_embed_cpp, m) {
    m.doc() = "C++ port of brisc's PaCMAP / LocalMAP (brisc.embed; MIT, Wainberg & Arbabi)";
    m.def("embed", &run, "pcs"_a, "knn"_a, "knn_sq_dist"_a, "out"_a, "localmap"_a,
          "num_neighbors"_a, "num_mid_near_pairs"_a, "num_further_pairs"_a, "it1"_a, "it2"_a,
          "it3"_a, "learning_rate"_a, "max_distance"_a, "seed"_a, "n_threads"_a = 0,
          "neighbor_pairs"_a.none() = nb::none(), "mid_near_pairs"_a.none() = nb::none(),
          "further_pairs"_a.none() = nb::none(),
          "PaCMAP (localmap=False) or LocalMAP (localmap=True) embedding of `pcs` into "
          "`out` (n x 2), identical to brisc's parallel code path. `knn` / `knn_sq_dist` "
          "are the nearest neighbours (excluding self) and squared Euclidean distances. "
          "The result does not depend on n_threads (0 = all OpenMP threads).");
}
