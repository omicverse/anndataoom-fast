#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <omp.h>

#include "kernels.hpp"
#include "knn.hpp"
#include "fuzzy.hpp"
#include "umap.hpp"
#include "groupstats.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace aof;

namespace {

template <class T>
using In1 = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
template <class T>
using Out1 = nb::ndarray<nb::numpy, T, nb::ndim<1>>;
using D2In = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using D2Out = nb::ndarray<nb::numpy, double, nb::ndim<2>>;
using F2In = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template <class T>
nb::ndarray<nb::numpy, T, nb::ndim<2>> np2t(std::vector<T>&& v, size_t r, size_t c) {
    auto* p = new std::vector<T>(std::move(v));
    nb::capsule owner(p, [](void* x) noexcept { delete static_cast<std::vector<T>*>(x); });
    return nb::ndarray<nb::numpy, T, nb::ndim<2>>(p->data(), {r, c}, owner);
}

template <class T>
std::vector<T> vec(const In1<T>& a) {
    return std::vector<T>(a.data(), a.data() + a.shape(0));
}

template <class T>
Out1<T> np(std::vector<T>&& v) {
    auto* p = new std::vector<T>(std::move(v));
    nb::capsule owner(p, [](void* x) noexcept { delete static_cast<std::vector<T>*>(x); });
    return Out1<T>(p->data(), {p->size()}, owner);
}

D2Out np2(std::vector<double>&& v, size_t r, size_t c) {
    auto* p = new std::vector<double>(std::move(v));
    nb::capsule owner(p, [](void* x) noexcept { delete static_cast<std::vector<double>*>(x); });
    return D2Out(p->data(), {r, c}, owner);
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "Multithreaded C++ kernels for out-of-core single-cell preprocessing";

    m.def("set_num_threads", [](int n) {
        if (n < 1) throw std::invalid_argument("n must be >= 1");
        omp_set_num_threads(n);
    });
    m.def("get_num_threads", []() { return omp_get_max_threads(); });

    nb::class_<Dataset>(m, "Dataset")
        .def(nb::new_([](const std::string& path, int dtype, int64_t n, int64_t offset) {
                 return std::make_shared<Dataset>(path, (DType)dtype, n, offset);
             }),
             "path"_a, "dtype"_a, "n_elems"_a, "offset"_a)
        .def(nb::new_([](const std::string& path, int dtype, int64_t n, int64_t chunk_elems,
                         In1<int64_t> addr, In1<int64_t> nbytes, In1<uint32_t> mask,
                         std::vector<int> filters) {
                 return std::make_shared<Dataset>(path, (DType)dtype, n, chunk_elems, vec(addr),
                                                  vec(nbytes), vec(mask), std::move(filters));
             }),
             "path"_a, "dtype"_a, "n_elems"_a, "chunk_elems"_a, "chunk_addr"_a,
             "chunk_nbytes"_a, "chunk_mask"_a, "filters"_a)
        .def_static("from_buffer", [](nb::ndarray<nb::ndim<1>, nb::c_contig, nb::device::cpu> a, int dtype) {
                 // The Python caller keeps `a` alive for the Dataset's lifetime.
                 return std::make_shared<Dataset>(a.data(), (DType)dtype, (int64_t)a.shape(0));
             }, "array"_a, "dtype"_a)
        .def_prop_ro("size", &Dataset::size)
        .def_prop_ro("chunked", &Dataset::chunked)
        .def("read", [](const Dataset& d, int64_t e0, int64_t e1) {
            std::vector<double> out((size_t)std::max<int64_t>(e1 - e0, 0));
            {
                nb::gil_scoped_release g;
                ReadScratch s;
                d.read_as<double>(e0, e1, out.data(), s);
            }
            return np(std::move(out));
        });

    nb::class_<CSRSource>(m, "CSRSource")
        .def(nb::new_([](std::shared_ptr<Dataset> data, std::shared_ptr<Dataset> indices,
                         In1<int64_t> indptr, int64_t n_cols, int64_t target_block_elems) {
                 return std::make_shared<CSRSource>(data, indices, vec(indptr), n_cols,
                                                    target_block_elems);
             }),
             "data"_a, "indices"_a, "indptr"_a, "n_cols"_a, "target_block_elems"_a = 1 << 21)
        .def_ro("n_rows", &CSRSource::n_rows)
        .def_ro("n_cols", &CSRSource::n_cols)
        .def_prop_ro("n_blocks", &CSRSource::n_blocks);

    m.def("qc_pass", [](const CSRSource& src, In1<uint8_t> gene_group,
                        std::vector<std::tuple<int, int, double>> rules,
                        std::optional<In1<uint8_t>> row_keep) {
        std::vector<QcRule> rr;
        for (auto& [s, o, t] : rules) rr.push_back({s, o, t});
        auto gg = vec(gene_group);
        std::vector<uint8_t> rk;
        if (row_keep) rk = vec(*row_keep);
        QcResult r;
        {
            nb::gil_scoped_release g;
            r = qc_pass(src, gg, rr, rk);
        }
        nb::dict d;
        d["total"] = np(std::move(r.total));
        d["nnz"] = np(std::move(r.nnz));
        d["mt"] = np(std::move(r.mt));
        d["ribo"] = np(std::move(r.ribo));
        d["hb"] = np(std::move(r.hb));
        d["mt_sum"] = np(std::move(r.mt_sum));
        d["ribo_sum"] = np(std::move(r.ribo_sum));
        d["hb_sum"] = np(std::move(r.hb_sum));
        d["nnz_pos"] = np(std::move(r.nnz_pos));
        d["keep"] = np(std::move(r.keep));
        d["g_stored"] = np(std::move(r.g_stored));
        d["g_nonzero"] = np(std::move(r.g_nonzero));
        d["g_sum"] = np(std::move(r.g_sum));
        d["g_sumsq"] = np(std::move(r.g_sumsq));
        return d;
    }, "src"_a, "gene_group"_a, "rules"_a, "row_keep"_a = nb::none());

    m.def("hi_pass", [](const CSRSource& src, In1<uint8_t> row_keep, In1<uint8_t> col_keep,
                        double max_fraction) {
        auto rk = vec(row_keep), ck = vec(col_keep);
        HiResult r;
        {
            nb::gil_scoped_release g;
            r = hi_pass(src, rk, ck, max_fraction);
        }
        nb::dict d;
        d["hi_count"] = np(std::move(r.hi_count));
        d["g_sum"] = np(std::move(r.g_sum));
        d["g_sumsq"] = np(std::move(r.g_sumsq));
        d["rowsum"] = np(std::move(r.rowsum));
        return d;
    }, "src"_a, "row_keep"_a, "col_keep"_a, "max_fraction"_a);

    m.def("stats_pass", [](const CSRSource& src, In1<uint8_t> row_keep, In1<int32_t> col_map,
                           In1<uint8_t> excl, In1<double> sums_genes, double target_sum,
                           bool log1p, bool do_scale, bool do_pearson, double theta, double clip,
                           double sum_total, std::optional<In1<double>> cells_override,
                           std::optional<In1<float>> recip_override, bool mu_f32_product) {
        StatsParams p;
        p.target_sum = target_sum;
        p.log1p = log1p;
        p.do_scale = do_scale;
        p.do_pearson = do_pearson;
        p.theta = theta;
        p.clip = clip;
        p.sum_total = sum_total;
        p.mu_f32_product = mu_f32_product;
        if (cells_override) p.cells_override = vec(*cells_override);
        if (recip_override) p.recip_override = vec(*recip_override);
        auto rk = vec(row_keep);
        auto cm = vec(col_map);
        auto ex = vec(excl);
        auto sg = vec(sums_genes);
        StatsResult r;
        {
            nb::gil_scoped_release g;
            r = stats_pass(src, rk, cm, ex, sg, p);
        }
        nb::dict d;
        d["cps"] = np(std::move(r.cps));
        d["nf"] = np(std::move(r.nf));
        d["recip"] = np(std::move(r.recip));
        d["s1"] = np(std::move(r.s1));
        d["s2"] = np(std::move(r.s2));
        d["d1"] = np(std::move(r.d1));
        d["d2"] = np(std::move(r.d2));
        d["g_sum"] = np(std::move(r.g_sum));
        d["g_sumsq"] = np(std::move(r.g_sumsq));
        return d;
    }, "src"_a, "row_keep"_a, "col_map"_a, "excl"_a, "sums_genes"_a, "target_sum"_a,
       "log1p"_a, "do_scale"_a, "do_pearson"_a, "theta"_a, "clip"_a, "sum_total"_a,
       "cells_override"_a = nb::none(), "recip_override"_a = nb::none(), "mu_f32_product"_a = false);

    m.def("pearson_zero_terms", [](In1<double> s_unique, In1<double> mult, In1<double> sums_genes,
                                   double sum_total, double theta, double clip, bool mu_f32_product) {
        auto su = vec(s_unique), mu = vec(mult), sg = vec(sums_genes);
        std::vector<double> z1, z2;
        {
            nb::gil_scoped_release g;
            pearson_zero_terms(su, mu, sg, sum_total, theta, clip, z1, z2, mu_f32_product);
        }
        return nb::make_tuple(np(std::move(z1)), np(std::move(z2)));
    }, "s_unique"_a, "mult"_a, "sums_genes"_a, "sum_total"_a, "theta"_a, "clip"_a,
       "mu_f32_product"_a = false);

    nb::class_<ScaledOperator>(m, "ScaledOperator")
        .def_prop_ro("shape", [](const ScaledOperator& o) {
            return nb::make_tuple(o.n_rows(), o.n_cols());
        })
        .def_prop_ro("nnz", &ScaledOperator::nnz)
        .def("matmul", [](const ScaledOperator& o, D2In Q) {
            if ((int64_t)Q.shape(0) != o.n_cols()) throw std::invalid_argument("shape mismatch");
            const int64_t k = Q.shape(1);
            std::vector<double> Y((size_t)o.n_rows() * k);
            {
                nb::gil_scoped_release g;
                o.matmul(Q.data(), k, Y.data());
            }
            return np2(std::move(Y), o.n_rows(), k);
        }, "Q"_a, "A @ Q for a float64 (n_cols, k) C-contiguous Q.")
        .def("rmatmul", [](const ScaledOperator& o, D2In P) {
            if ((int64_t)P.shape(0) != o.n_rows()) throw std::invalid_argument("shape mismatch");
            const int64_t k = P.shape(1);
            std::vector<double> Z((size_t)o.n_cols() * k);
            {
                nb::gil_scoped_release g;
                o.rmatmul(P.data(), k, Z.data());
            }
            return np2(std::move(Z), o.n_cols(), k);
        }, "P"_a, "A.T @ P for a float64 (n_rows, k) C-contiguous P.")
        .def("csr", [](const ScaledOperator& o) {
            const ScaledCSR& c = o.csr();
            return nb::make_tuple(np(std::vector<int64_t>(c.indptr)),
                                  np(std::vector<int32_t>(c.indices)),
                                  np(std::vector<float>(c.data)), np(std::vector<float>(c.z)));
        });

    m.def("orthonormalize", [](D2In Y) {
        const int64_t n = Y.shape(0), k = Y.shape(1);
        std::vector<double> out(Y.data(), Y.data() + (size_t)n * k);
        {
            nb::gil_scoped_release g;
            orthonormalize(out.data(), n, k);
        }
        return np2(std::move(out), n, k);
    }, "Y"_a, "Orthonormal basis of span(Y) by deterministic CholeskyQR2.");

    m.def("gemm_tall", [](D2In A, D2In B) {
        const int64_t n = A.shape(0), k = A.shape(1), mm = B.shape(1);
        if ((int64_t)B.shape(0) != k) throw std::invalid_argument("shape mismatch");
        std::vector<double> C((size_t)n * mm);
        {
            nb::gil_scoped_release g;
            gemm_tall(A.data(), B.data(), n, k, mm, C.data());
        }
        return np2(std::move(C), n, mm);
    }, "A"_a, "B"_a, "A @ B for tall A, row-parallel and deterministic.");

    m.def("knn_ivf", [](F2In X, int64_t k, In1<int64_t> init, int64_t n_probe, int kmeans_iters,
                        int64_t block) {
        KnnParams p;
        p.k = k;
        p.n_probe = n_probe;
        p.kmeans_iters = kmeans_iters;
        p.block = block;
        auto ini = vec(init);
        const int64_t n = X.shape(0), d = X.shape(1);
        KnnResult r;
        {
            nb::gil_scoped_release g;
            r = knn_ivf(X.data(), n, d, p, ini);
        }
        const size_t kk = (size_t)std::min<int64_t>(k, n - 1);
        return nb::make_tuple(np2t(std::move(r.indices), n, kk), np2t(std::move(r.distances), n, kk),
                              r.n_clusters, r.n_probe);
    }, "X"_a, "k"_a, "init"_a, "n_probe"_a, "kmeans_iters"_a, "block"_a = 256);

    m.def("fuzzy_simplicial_set", [](nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> idx,
                                     F2In dist, double n_neighbors, double local_connectivity,
                                     double set_op_mix_ratio, double bandwidth) {
        const int64_t n = idx.shape(0), k = idx.shape(1);
        if ((int64_t)dist.shape(0) != n || (int64_t)dist.shape(1) != k)
            throw std::invalid_argument("knn_indices and knn_dists differ in shape");
        FuzzyResult r;
        {
            nb::gil_scoped_release g;
            r = fuzzy_simplicial_set(idx.data(), dist.data(), n, k, n_neighbors, local_connectivity,
                                     set_op_mix_ratio, bandwidth);
        }
        return nb::make_tuple(np(std::move(r.indptr)), np(std::move(r.indices)), np(std::move(r.data)),
                              np(std::move(r.sigmas)), np(std::move(r.rhos)));
    }, "knn_indices"_a, "knn_dists"_a, "n_neighbors"_a, "local_connectivity"_a = 1.0,
       "set_op_mix_ratio"_a = 1.0, "bandwidth"_a = 1.0);

    m.def("umap_optimize_euclidean", [](nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> emb,
                                        In1<int32_t> head, In1<int32_t> tail, In1<double> eps,
                                        In1<int64_t> rng_state, double a, double b, double gamma,
                                        double initial_alpha, double negative_sample_rate, int n_epochs,
                                        bool move_other, bool parallel) {
        if (rng_state.shape(0) != 3) throw std::invalid_argument("rng_state must have 3 entries");
        const int64_t E = head.shape(0);
        if ((int64_t)tail.shape(0) != E || (int64_t)eps.shape(0) != E)
            throw std::invalid_argument("head/tail/epochs_per_sample length mismatch");
        UmapSgdParams p;
        p.a = a; p.b = b; p.gamma = gamma; p.initial_alpha = initial_alpha;
        p.negative_sample_rate = negative_sample_rate; p.n_epochs = n_epochs;
        p.move_other = move_other; p.parallel = parallel;
        nb::gil_scoped_release g;
        umap_optimize_euclidean(emb.data(), emb.shape(0), (int)emb.shape(1), head.data(), tail.data(), E,
                                eps.data(), rng_state.data(), p);
    }, "embedding"_a, "head"_a, "tail"_a, "epochs_per_sample"_a, "rng_state"_a, "a"_a, "b"_a,
       "gamma"_a, "initial_alpha"_a, "negative_sample_rate"_a, "n_epochs"_a, "move_other"_a = true,
       "parallel"_a = false);

    m.def("umap_optimize_euclidean_det", [](nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> emb,
                                            In1<int64_t> indptr, In1<int32_t> tail, In1<double> eps,
                                            uint64_t seed, double a, double b, double gamma,
                                            double initial_alpha, double negative_sample_rate,
                                            int n_epochs, bool move_other, int64_t block) {
        const int64_t n = emb.shape(0);
        if ((int64_t)indptr.shape(0) != n + 1) throw std::invalid_argument("indptr must have n+1 entries");
        const int64_t E = indptr.data()[n];
        if ((int64_t)tail.shape(0) != E || (int64_t)eps.shape(0) != E)
            throw std::invalid_argument("tail/epochs_per_sample length mismatch");
        UmapSgdParams p;
        p.a = a; p.b = b; p.gamma = gamma; p.initial_alpha = initial_alpha;
        p.negative_sample_rate = negative_sample_rate; p.n_epochs = n_epochs;
        p.move_other = move_other;
        nb::gil_scoped_release g;
        umap_optimize_euclidean_det(emb.data(), n, (int)emb.shape(1), indptr.data(), tail.data(),
                                    eps.data(), seed, p, block);
    }, "embedding"_a, "indptr"_a, "tail"_a, "epochs_per_sample"_a, "seed"_a, "a"_a, "b"_a,
       "gamma"_a, "initial_alpha"_a, "negative_sample_rate"_a, "n_epochs"_a, "move_other"_a = true,
       "block"_a = 0);

    m.def("csr_matvec", [](In1<int64_t> indptr, In1<int32_t> indices, In1<double> data, In1<double> x) {
        const int64_t n = indptr.shape(0) - 1;
        std::vector<double> y(n);
        {
            nb::gil_scoped_release g;
            csr_matvec(indptr.data(), indices.data(), data.data(), n, x.data(), y.data());
        }
        return np(std::move(y));
    }, "indptr"_a, "indices"_a, "data"_a, "x"_a);

    m.def("scale_dense", [](const CSRSource& src, In1<uint8_t> row_keep, In1<int32_t> col_map,
                            In1<double> mean, In1<double> sd, double clip,
                            nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu> out) {
        auto rk = vec(row_keep);
        auto cm = vec(col_map);
        auto mu = vec(mean), s = vec(sd);
        int64_t kept = 0;
        for (auto v : rk) kept += v != 0;
        if ((int64_t)out.shape(0) != kept || (int64_t)out.shape(1) != (int64_t)mu.size())
            throw std::invalid_argument("out has the wrong shape");
        nb::gil_scoped_release g;
        scale_dense(src, rk, cm, mu, s, clip, out.data());
    }, "src"_a, "row_keep"_a, "col_map"_a, "mean"_a, "std"_a, "clip"_a, "out"_a);

    m.def("group_pass", [](const CSRSource& src, In1<int32_t> group, int64_t n_groups,
                           In1<int32_t> col_map, In1<float> recip, bool log1p) {
        auto gr = vec(group);
        auto cm = vec(col_map);
        auto rc = vec(recip);
        int64_t K = 0;
        for (int32_t v : cm) K = std::max<int64_t>(K, (int64_t)v + 1);
        GroupPassResult r;
        {
            nb::gil_scoped_release g;
            r = group_pass(src, gr, n_groups, cm, rc, log1p);
        }
        return nb::make_tuple(np2t(std::move(r.sum), n_groups, K), np2t(std::move(r.sumsq), n_groups, K),
                              np2t(std::move(r.nnz), n_groups, K));
    }, "src"_a, "group"_a, "n_groups"_a, "col_map"_a, "recip"_a, "log1p"_a);

    auto bind_group_stats = [&](auto vtag, auto itag) {
        using V = decltype(vtag);
        using I = decltype(itag);
        m.def("group_stats_csr", [](In1<int64_t> indptr, In1<I> indices, In1<V> data, int64_t n_cols,
                                    In1<int64_t> order, In1<int64_t> offsets) {
            const int64_t C = offsets.shape(0) - 1;
            GroupStats r;
            {
                nb::gil_scoped_release g;
                r = group_stats_csr<V, I>(indptr.data(), indices.data(), data.data(), n_cols,
                                          order.data(), offsets.data(), C);
            }
            return nb::make_tuple(np2t(std::move(r.sum), C, n_cols), np2t(std::move(r.sumsq), C, n_cols),
                                  np2t(std::move(r.nnz), C, n_cols));
        }, "indptr"_a, "indices"_a, "data"_a, "n_cols"_a, "order"_a, "offsets"_a);
    };
    bind_group_stats(float{}, int32_t{});
    bind_group_stats(float{}, int64_t{});
    bind_group_stats(double{}, int32_t{});
    bind_group_stats(double{}, int64_t{});

    m.def("extract_scaled", [](const CSRSource& src, In1<uint8_t> row_keep, In1<int32_t> col_map,
                               In1<double> mean, In1<double> sd, In1<float> recip, bool log1p,
                               double clip) {
        auto rk = vec(row_keep);
        auto cm = vec(col_map);
        auto mu = vec(mean), s = vec(sd);
        auto rc = vec(recip);
        ScaledOperator* op;
        {
            nb::gil_scoped_release g;
            op = new ScaledOperator(extract_scaled(src, rk, cm, mu, s, rc, log1p, clip));
        }
        return op;
    }, nb::rv_policy::take_ownership, "src"_a, "row_keep"_a, "col_map"_a, "mean"_a, "std"_a,
       "recip"_a, "log1p"_a, "clip"_a);
}
