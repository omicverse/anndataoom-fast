// nanobind bindings for the exact leidenalg port (module anndataoom_fast._leiden_cpp).
//
// The algorithm is ported from leidenalg / libleidenalg (V.A. Traag,
// GNU GPL-3.0-or-later) and the igraph C core (igraph development team,
// GNU GPL-2.0-or-later); see leiden.cpp.  This file is part of
// anndataoom-fast, GNU GPL-3.0-or-later.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <omp.h>

#include <cmath>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

#include "leiden.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using I64In = nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using F64In = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using I64Out = nb::ndarray<nb::numpy, int64_t, nb::ndim<1>>;

I64Out to_numpy(std::vector<int64_t>&& v) {
    auto* p = new std::vector<int64_t>(std::move(v));
    nb::capsule owner(p, [](void* x) noexcept { delete static_cast<std::vector<int64_t>*>(x); });
    return I64Out(p->data(), {p->size()}, owner);
}

std::unique_ptr<aof_leiden::Graph> build_graph(size_t n, const I64In& src, const I64In& dst,
                                                const F64In* weights, bool directed) {
    const size_t m = src.shape(0);
    if (dst.shape(0) != m) throw std::invalid_argument("src and dst must have the same length");
    if (n >= size_t(UINT32_MAX)) throw std::invalid_argument("too many vertices");
    if (m >= size_t(UINT32_MAX)) throw std::invalid_argument("too many edges");
    std::vector<uint32_t> a(m), b(m);
    const int64_t* s = src.data();
    const int64_t* d = dst.data();
    for (size_t e = 0; e < m; e++) {
        if (s[e] < 0 || d[e] < 0 || size_t(s[e]) >= n || size_t(d[e]) >= n)
            throw std::invalid_argument("vertex id out of range");
        a[e] = uint32_t(s[e]);
        b[e] = uint32_t(d[e]);
    }
    std::vector<double> w;
    if (weights) {
        if (weights->shape(0) != m)
            throw std::invalid_argument("Weight vector not the same size as the number of edges.");
        const double* p = weights->data();
        w.assign(p, p + m);
        // create_graph_from_py(..., check_positive_weight=true)
        for (size_t e = 0; e < m; e++) {
            if (w[e] < 0) throw std::invalid_argument("Cannot accept negative weights.");
            if (std::isnan(w[e])) throw std::invalid_argument("Cannot accept NaN weights.");
            if (!std::isfinite(w[e])) throw std::invalid_argument("Cannot accept infinite weights.");
        }
    } else {
        w.assign(m, 1.0);
    }
    return std::unique_ptr<aof_leiden::Graph>(
        aof_leiden::Graph::create(n, directed, a, b, std::move(w), {}));
}

}  // namespace

NB_MODULE(_leiden_cpp, m) {
    m.doc() =
        "Exact port of leidenalg.find_partition(RBConfigurationVertexPartition) "
        "(leidenalg 0.12 / libleidenalg 0.13 / igraph 1.0)";

    m.def(
        "find_partition",
        [](size_t n, I64In src, I64In dst, std::optional<F64In> weights, bool directed,
           std::vector<double> resolutions, uint64_t seed, int n_iterations, int n_threads) {
            std::vector<std::vector<int64_t>> out(resolutions.size());
            {
                nb::gil_scoped_release g;
                auto graph = build_graph(n, src, dst, weights ? &*weights : nullptr, directed);
                aof_leiden::Options opt;
                opt.n_iterations = n_iterations;
                const int nres = int(resolutions.size());
                const int total = n_threads > 0 ? n_threads : omp_get_max_threads();
                // one outer thread per resolution; the remaining threads are
                // used (nested) for graph aggregation inside each run.
                int nt = total < nres ? total : nres;
                if (nt < 1) nt = 1;
                opt.n_threads = total / nt > 1 ? total / nt : 1;
                const int saved_levels = omp_get_max_active_levels();
                if (opt.n_threads > 1 && nt > 1 && saved_levels < 2) omp_set_max_active_levels(2);
                std::vector<std::string> errors(resolutions.size());
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
                for (int i = 0; i < nres; i++) {
                    try {
                        out[i] = aof_leiden::find_partition(*graph, resolutions[i], seed, opt);
                    } catch (const std::exception& e) {
                        errors[i] = e.what();
                        if (errors[i].empty()) errors[i] = "unknown error";
                    }
                }
                if (omp_get_max_active_levels() != saved_levels) omp_set_max_active_levels(saved_levels);
                for (auto& e : errors)
                    if (!e.empty()) throw std::runtime_error(e);
            }
            nb::list res;
            for (auto& v : out) res.append(to_numpy(std::move(v)));
            return res;
        },
        "n"_a, "src"_a, "dst"_a, "weights"_a.none(), "directed"_a, "resolutions"_a, "seed"_a,
        "n_iterations"_a, "n_threads"_a = 0,
        "Run leidenalg's RBConfiguration Leiden for each resolution (one OpenMP thread "
        "per resolution, each with its own identically seeded RNG; spare threads are "
        "used for graph aggregation). n_threads=0 uses all OpenMP threads. Results do "
        "not depend on the number of threads. Returns a list of int64 membership arrays.");
}
