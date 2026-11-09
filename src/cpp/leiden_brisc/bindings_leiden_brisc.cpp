// nanobind bindings for the brisc Leiden port (module
// anndataoom_fast._leiden_brisc_cpp).
//
// The algorithm is ported from brisc (brisc/leiden.pyx), Copyright (c) 2026
// Michael Wainberg and Keon Arbabi, MIT License; see leiden_brisc.cpp.  This
// file is part of anndataoom-fast, GNU GPL-3.0-or-later.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <omp.h>

#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "leiden_brisc.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using F32In = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using U32In = nb::ndarray<const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using I64In = nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using U32Out = nb::ndarray<uint32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template <typename Idx>
std::vector<uint32_t> run(F32In data, nb::ndarray<const Idx, nb::ndim<1>, nb::c_contig, nb::device::cpu> indices,
                          nb::ndarray<const Idx, nb::ndim<1>, nb::c_contig, nb::device::cpu> indptr,
                          U32In nbr, std::optional<I64In> nbr_indptr, uint64_t nbr_k,
                          U32Out out, std::vector<float> resolutions,
                          uint32_t min_cluster_size, uint64_t seed, int n_threads) {
    if (indptr.shape(0) < 2) throw std::invalid_argument("graph has no nodes");
    const uint64_t n = indptr.shape(0) - 1;
    if (n >= uint64_t(UINT32_MAX)) throw std::invalid_argument("too many nodes (must be < 2^32 - 1)");
    const size_t nres = resolutions.size();
    if (out.shape(0) != nres || out.shape(1) != n)
        throw std::invalid_argument("out must have shape (len(resolutions), n)");
    const Idx* ip = indptr.data();
    const Idx nnz = ip[n];
    if (ip[0] != 0 || nnz < 0 || size_t(nnz) != indices.shape(0) || size_t(nnz) != data.shape(0))
        throw std::invalid_argument("inconsistent CSR arrays");
    for (uint64_t r = 0; r < n; r++)
        if (ip[r + 1] < ip[r]) throw std::invalid_argument("indptr must be non-decreasing");
    const Idx* ix = indices.data();
    for (Idx e = 0; e < nnz; e++)
        if (ix[e] < 0 || uint64_t(ix[e]) >= n)
            throw std::invalid_argument("column index out of range");

    aof_brisc::Neighbors nb_;
    nb_.nbr = nbr.data();
    nb_.k = nbr_k;
    if (nbr_indptr) {
        if (nbr_indptr->shape(0) != n + 1) throw std::invalid_argument("bad neighbors indptr");
        nb_.indptr = nbr_indptr->data();
        if (nb_.indptr[0] != 0 || uint64_t(nb_.indptr[n]) > nbr.shape(0))
            throw std::invalid_argument("bad neighbors indptr");
        for (uint64_t r = 0; r < n; r++)
            if (nb_.indptr[r + 1] < nb_.indptr[r])
                throw std::invalid_argument("bad neighbors indptr");
    } else if (nbr.shape(0) != n * nbr_k) {
        throw std::invalid_argument("neighbors must have n * k entries");
    }

    std::vector<uint32_t> num_final(nres, 0);
    std::vector<std::string> errors(nres);
    {
        nb::gil_scoped_release g;
        const int total = n_threads > 0 ? n_threads : omp_get_max_threads();
        int nt = total < int(nres) ? total : int(nres);
        if (nt < 1) nt = 1;
        const float* d = data.data();
        uint32_t* o = out.data();
        const bool multi = nres > 1;
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
        for (int r = 0; r < int(nres); r++) {
            try {
                num_final[size_t(r)] = aof_brisc::leiden<Idx>(
                    d, ix, ip, n, nb_, o + size_t(r) * n, resolutions[size_t(r)],
                    min_cluster_size, seed, multi);
            } catch (const std::exception& e) {
                errors[size_t(r)] = e.what();
                if (errors[size_t(r)].empty()) errors[size_t(r)] = "unknown error";
            } catch (...) {
                errors[size_t(r)] = "unknown error";
            }
        }
    }
    for (auto& e : errors)
        if (!e.empty()) throw std::invalid_argument(e);
    return num_final;
}

}  // namespace

NB_MODULE(_leiden_brisc_cpp, m) {
    m.doc() = "C++ port of brisc's Leiden clustering (brisc.leiden; MIT, Wainberg & Arbabi)";

    const char* doc =
        "Run brisc's Leiden for each resolution (one OpenMP thread per resolution, "
        "each an independent run seeded with `seed`), writing labels into out[r]. "
        "n_threads=0 uses all OpenMP threads. Results do not depend on the number of "
        "threads. Returns the number of clusters per resolution.";
    m.def("leiden", &run<int32_t>, "data"_a, "indices"_a, "indptr"_a, "nbr"_a,
          "nbr_indptr"_a.none(), "nbr_k"_a, "out"_a, "resolutions"_a, "min_cluster_size"_a,
          "seed"_a, "n_threads"_a = 0, doc);
    m.def("leiden", &run<int64_t>, "data"_a, "indices"_a, "indptr"_a, "nbr"_a,
          "nbr_indptr"_a.none(), "nbr_k"_a, "out"_a, "resolutions"_a, "min_cluster_size"_a,
          "seed"_a, "n_threads"_a = 0, doc);
}
