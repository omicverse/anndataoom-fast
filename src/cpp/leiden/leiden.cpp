// Exact C++ port of leidenalg (libleidenalg 0.13.0) RBConfigurationVertexPartition
// optimisation, including the pieces of the igraph C core 1.0 it relies on.
//
// Ported from:
//   * libleidenalg / leidenalg by V.A. Traag, GNU GPL-3.0-or-later:
//     GraphHelper.cpp, MutableVertexPartition.cpp,
//     RBConfigurationVertexPartition.cpp, Optimiser.cpp, Optimiser.py,
//     functions.py, python_partition_interface.cpp.
//   * igraph C core by the igraph development team, GNU GPL-2.0-or-later:
//     rng_mt19937.c, random.c (igraph_rng_get_integer & helpers), vector.c
//     (igraph_vector_int_pair_order), type_indexededgelist.c
//     (igraph_add_edges, igraph_neighbors, igraph_incident, igraph_get_eid).
//
// Every floating point operation is performed in the same order and precision
// as in libleidenalg, and every random number is drawn in the same order from
// an identical MT19937 stream, so memberships are identical element by element.
// Only the data structures differ: igraph lazy lists are replaced by flat
// arrays built once per (aggregated) graph, preserving their iteration order.
//
// This file is part of anndataoom-fast, GNU GPL-3.0-or-later.

#include "leiden.hpp"

#if defined(_MSC_VER) && !defined(__clang__)
#define __builtin_prefetch(...) ((void)0)
#endif

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

namespace aof_leiden {

// ---------------------------------------------------------------------------
// igraph RNG: MT19937 (rng_mt19937.c) + igraph_rng_get_integer (random.c)
// ---------------------------------------------------------------------------
namespace {

class RNG {
    static constexpr int N = 624, M = 397;
    uint32_t mt[N];
    int mti;

public:
    RNG() { seed(0); }

    void seed(uint64_t s) {
        std::memset(mt, 0, sizeof(mt));
        if (s == 0) s = 4357;
        mt[0] = uint32_t(s & UINT32_C(0xffffffff));
        int i;
        for (i = 1; i < N; i++) {
            mt[i] = (UINT32_C(1812433253) * (mt[i - 1] ^ (mt[i - 1] >> 30)) + uint32_t(i));
        }
        mti = i;
    }

    uint32_t get() {
        static const uint32_t UPPER = UINT32_C(0x80000000), LOWER = UINT32_C(0x7fffffff);
#define AOF_MAGIC(y) (((y) & 0x1) ? UINT32_C(0x9908b0df) : 0)
        if (mti >= N) {
            int kk;
            for (kk = 0; kk < N - M; kk++) {
                uint32_t y = (mt[kk] & UPPER) | (mt[kk + 1] & LOWER);
                mt[kk] = mt[kk + M] ^ (y >> 1) ^ AOF_MAGIC(y);
            }
            for (; kk < N - 1; kk++) {
                uint32_t y = (mt[kk] & UPPER) | (mt[kk + 1] & LOWER);
                mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ AOF_MAGIC(y);
            }
            uint32_t y = (mt[N - 1] & UPPER) | (mt[0] & LOWER);
            mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ AOF_MAGIC(y);
            mti = 0;
        }
#undef AOF_MAGIC
        uint32_t k = mt[mti];
        k ^= (k >> 11);
        k ^= (k << 7) & UINT32_C(0x9d2c5680);
        k ^= (k << 15) & UINT32_C(0xefc60000);
        k ^= (k >> 18);
        mti++;
        return k;
    }

    // igraph_i_rng_get_uint32_bounded (Lemire, debiased)
    uint32_t bounded32(uint32_t range) {
        uint32_t x, l, t = (-range) % range;
        uint64_t mm;
        do {
            x = get();
            mm = uint64_t(x) * uint64_t(range);
            l = uint32_t(mm);
        } while (l < t);
        return uint32_t(mm >> 32);
    }

    // igraph_i_rng_get_uint64_bounded; 64 random bits = (get() << 32) + get()
    uint64_t bounded64(uint64_t range) {
        uint64_t x, l, t = (-range) % range;
        __uint128_t mm;
        do {
            uint64_t hi = get();
            uint64_t lo = get();
            x = (hi << 32) + lo;
            mm = __uint128_t(x) * __uint128_t(range);
            l = uint64_t(mm);
        } while (l < t);
        return uint64_t(mm >> 64);
    }

    // igraph_rng_get_integer(rng, l, h) for 0 <= l <= h
    size_t integer(size_t l, size_t h) {
        if (h == l) return l;
        uint64_t range = uint64_t(h - l) + 1;
        if (range <= UINT32_MAX) return l + bounded32(uint32_t(range));
        return l + size_t(bounded64(range));
    }
};

// GraphHelper.cpp shuffle()
void shuffle(std::vector<uint32_t>& v, RNG& rng) {
    size_t n = v.size();
    if (n > 0) {
        for (size_t idx = n - 1; idx > 0; idx--) {
            size_t r = rng.integer(0, idx);
            uint32_t tmp = v[idx];
            v[idx] = v[r];
            v[r] = tmp;
        }
    }
}

// Equivalent of igraph_vector_int_pair_order(v, v2, res, nodes).  igraph's
// two-pass linked-list radix sort orders edge ids by v, then by v2, and
// multi-edges (equal v and v2) by *decreasing* edge id; the same order is
// produced here by two stable counting sorts (the first one visiting edge
// ids in decreasing order).
void pair_order(const std::vector<uint32_t>& v, const std::vector<uint32_t>& v2,
                std::vector<uint32_t>& res, size_t nodes) {
    const size_t edges = v.size();
    std::vector<size_t> cnt(nodes + 1, 0);
    std::vector<uint32_t> tmp(edges);
    res.resize(edges);
    for (size_t e = 0; e < edges; e++) cnt[v2[e] + 1]++;
    for (size_t i = 0; i < nodes; i++) cnt[i + 1] += cnt[i];
    for (size_t e = edges; e-- > 0;) tmp[cnt[v2[e]]++] = uint32_t(e);
    std::fill(cnt.begin(), cnt.end(), 0);
    for (size_t e = 0; e < edges; e++) cnt[v[e] + 1]++;
    for (size_t i = 0; i < nodes; i++) cnt[i + 1] += cnt[i];
    for (size_t i = 0; i < edges; i++) {
        uint32_t e = tmp[i];
        res[cnt[v[e]]++] = e;
    }
}

// Stable counting sort of items 0..m-1 by key(i) in [0, nkeys): calls
// emit(i, position).  starts receives the bucket offsets (nkeys + 1).  With
// several threads each thread handles a contiguous range of items with its
// own histogram; positions are identical to the sequential version.
template <class KeyF, class EmitF>
void stable_scatter(size_t m, size_t nkeys, int nt, KeyF key, EmitF emit,
                    std::vector<size_t>& starts) {
    starts.assign(nkeys + 1, 0);
    if (nt <= 1 || m < 100000) {
        for (size_t i = 0; i < m; i++) starts[key(i) + 1]++;
        for (size_t k = 0; k < nkeys; k++) starts[k + 1] += starts[k];
        std::vector<size_t> pos(starts.begin(), starts.end() - 1);
        for (size_t i = 0; i < m; i++) emit(i, pos[key(i)]++);
        return;
    }
    std::vector<std::vector<size_t>> hist(nt);
#pragma omp parallel num_threads(nt)
    {
        const int t = omp_get_thread_num();
        const size_t b = m * size_t(t) / size_t(nt), e = m * size_t(t + 1) / size_t(nt);
        std::vector<size_t>& h = hist[t];
        h.assign(nkeys, 0);
        for (size_t i = b; i < e; i++) h[key(i)]++;
#pragma omp barrier
#pragma omp single
        {
            size_t run = 0;
            for (size_t k = 0; k < nkeys; k++) {
                starts[k] = run;
                for (int tt = 0; tt < nt; tt++) {
                    size_t c = hist[tt][k];
                    hist[tt][k] = run;
                    run += c;
                }
            }
            starts[nkeys] = run;
        }
        for (size_t i = b; i < e; i++) emit(i, h[key(i)]++);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Graph (igraph_t storage + leidenalg Graph admin)
// ---------------------------------------------------------------------------
Graph* Graph::create(size_t n, bool directed, const std::vector<uint32_t>& a,
                     const std::vector<uint32_t>& b, std::vector<double>&& weights,
                     std::vector<double>&& node_sizes) {
    std::unique_ptr<Graph> g(new Graph());
    if (n > size_t(ID_MASK)) throw std::invalid_argument("too many vertices");
    g->n = n;
    g->m = a.size();
    g->directed = directed;
    const size_t m = g->m;
    if (weights.size() != m) throw std::invalid_argument("weights size mismatch");
    g->ew = std::move(weights);
    if (node_sizes.empty()) node_sizes.assign(n, 1.0);
    if (node_sizes.size() != n) throw std::invalid_argument("node_sizes size mismatch");
    g->node_size = std::move(node_sizes);

    // igraph_add_edges: undirected edges stored as from = max, to = min.
    g->efrom.resize(m);
    g->eto.resize(m);
    for (size_t e = 0; e < m; e++) {
        if (a[e] >= n || b[e] >= n) throw std::invalid_argument("vertex id out of range");
        if (directed || a[e] > b[e]) {
            g->efrom[e] = a[e];
            g->eto[e] = b[e];
        } else {
            g->efrom[e] = b[e];
            g->eto[e] = a[e];
        }
    }
    const uint32_t* F = g->efrom.data();
    const uint32_t* T = g->eto.data();
    const double* W = g->ew.data();
    for (size_t e = 0; e < m; e++)
        if (!(W[e] > 0)) { g->all_positive = false; break; }

    std::vector<uint32_t> oi, ii;
    pair_order(g->efrom, g->eto, oi, n);
    pair_order(g->eto, g->efrom, ii, n);
    std::vector<size_t> os(n + 1, 0), is(n + 1, 0);
    for (size_t e = 0; e < m; e++) {
        os[F[e] + 1]++;
        is[T[e] + 1]++;
    }
    for (size_t v = 0; v < n; v++) {
        os[v + 1] += os[v];
        is[v + 1] += is[v];
    }

    auto& off = g->off;
    auto& nb = g->nb;
    auto& ww = g->w;
    off.assign(n + 1, 0);
    nb.resize(2 * m);
    ww.resize(2 * m);
    size_t idx = 0;
    if (directed) {
        // igraph_neighbors / igraph_incident (mode=ALL, directed): merge of the
        // out-list (sorted by target) and the in-list (sorted by source); on
        // equal neighbour ids one out-entry then one in-entry.
        for (size_t v = 0; v < n; v++) {
            off[v] = idx;
            size_t i1 = os[v], j1 = os[v + 1], i2 = is[v], j2 = is[v + 1];
            while (i1 < j1 && i2 < j2) {
                uint32_t e1 = oi[i1], e2 = ii[i2];
                uint32_t n1 = T[e1], n2 = F[e2];
                if (n1 < n2) {
                    i1++;
                    nb[idx] = n1; ww[idx++] = W[e1];
                } else if (n1 > n2) {
                    i2++;
                    nb[idx] = n2 | IN_FLAG; ww[idx++] = W[e2];
                } else {
                    i1++; i2++;
                    nb[idx] = n1; ww[idx++] = W[e1];
                    nb[idx] = n2 | IN_FLAG; ww[idx++] = W[e2];
                }
            }
            while (i1 < j1) {
                uint32_t e1 = oi[i1++];
                nb[idx] = T[e1]; ww[idx++] = W[e1];
            }
            while (i2 < j2) {
                uint32_t e2 = ii[i2++];
                nb[idx] = F[e2] | IN_FLAG; ww[idx++] = W[e2];
            }
        }
    } else {
        // Undirected: out part (stored from == v) then in part (stored to == v).
        g->outcnt.assign(n, 0);
        for (size_t v = 0; v < n; v++) {
            off[v] = idx;
            for (size_t i = os[v]; i < os[v + 1]; i++) {
                uint32_t e = oi[i];
                nb[idx] = T[e]; ww[idx++] = W[e];
            }
            g->outcnt[v] = uint32_t(os[v + 1] - os[v]);
            for (size_t i = is[v]; i < is[v + 1]; i++) {
                uint32_t e = ii[i];
                nb[idx] = F[e]; ww[idx++] = W[e];
            }
        }
    }
    off[n] = idx;
    if (directed) {
        g->out_off = os;
        g->out_nb.resize(m);
        g->out_w.resize(m);
        for (size_t i = 0; i < m; i++) {
            g->out_nb[i] = T[oi[i]];
            g->out_w[i] = W[oi[i]];
        }
    }

    // Graph::init_admin()
    g->strength_in.assign(n, 0.0);
    if (directed) g->strength_out.assign(n, 0.0);
    g->total_weight = 0.0;
    for (size_t e = 0; e < m; e++) {
        double w = W[e];
        g->total_weight += w;
        uint32_t from = F[e], to = T[e];
        if (directed) {
            g->strength_in[to] += w;
            g->strength_out[from] += w;
        } else {
            g->strength_in[to] += w;
            g->strength_in[from] += w;
        }
    }
    g->total_size = 0;
    for (size_t v = 0; v < n; v++) g->total_size += g->node_size[v];

    // Graph::set_self_weights(): igraph_get_eid(v, v) finds the first loop in
    // the sorted out- (or in-) list, i.e. the loop with the largest edge id.
    std::vector<int64_t> self_e(n, -1);
    for (size_t e = 0; e < m; e++)
        if (F[e] == T[e]) self_e[F[e]] = int64_t(e);
    g->self_weight.assign(n, 0.0);
    for (size_t v = 0; v < n; v++)
        if (self_e[v] >= 0) g->self_weight[v] = W[self_e[v]];

    return g.release();
}

Graph* Graph::create_grouped_directed(size_t n, std::vector<uint32_t>&& from,
                                     std::vector<uint32_t>&& to, std::vector<double>&& weights,
                                     std::vector<double>&& node_sizes, int nthreads) {
    std::unique_ptr<Graph> g(new Graph());
    if (n > size_t(ID_MASK)) throw std::invalid_argument("too many vertices");
    const size_t m = from.size();
    g->n = n;
    g->m = m;
    g->directed = true;
    g->efrom = std::move(from);
    g->eto = std::move(to);
    g->ew = std::move(weights);
    g->node_size = std::move(node_sizes);
    if (g->node_size.empty()) g->node_size.assign(n, 1.0);
    const uint32_t* F = g->efrom.data();
    const uint32_t* T = g->eto.data();
    const double* W = g->ew.data();
    const int nt = (nthreads > 1 && m > 50000) ? nthreads : 1;

    bool allpos = true;
    for (size_t e = 0; e < m; e++)
        if (!(W[e] > 0)) { allpos = false; break; }
    g->all_positive = allpos;

    // (from, to) pairs are distinct, so pair_order needs no tie-breaking:
    // ii = pair_order(to, from) is a stable counting sort of the ids by
    // target (ids increase with the source), and oi = pair_order(from, to) a
    // stable counting sort of ii by source.  The endpoint and the weight are
    // carried along instead of the edge id.
    std::vector<size_t> os, is;
    std::vector<uint32_t> in_from(m), out_to(m);
    std::vector<double> in_w(m), out_w(m);
    {
        uint32_t* IFp = in_from.data();
        double* IWp = in_w.data();
        stable_scatter(m, n, nt, [T](size_t e) { return T[e]; },
                       [F, W, IFp, IWp](size_t e, size_t p) {
                           IFp[p] = F[e];
                           IWp[p] = W[e];
                       },
                       is);
        uint32_t* OTp = out_to.data();
        double* OWp = out_w.data();
        if (nt <= 1) {
            os.assign(n + 1, 0);
            for (size_t k = 0; k < m; k++) os[IFp[k] + 1]++;
            for (size_t v = 0; v < n; v++) os[v + 1] += os[v];
            std::vector<size_t> pos(os.begin(), os.end() - 1);
            for (size_t u = 0; u < n; u++) {
                for (size_t k = is[u]; k < is[u + 1]; k++) {
                    size_t p = pos[IFp[k]]++;
                    OTp[p] = uint32_t(u);
                    OWp[p] = IWp[k];
                }
            }
        } else {
            // in-list position k belongs to target u with is[u] <= k < is[u+1]
            std::vector<uint32_t> in_to(m);
            uint32_t* ITp = in_to.data();
#pragma omp parallel for schedule(static) num_threads(nt)
            for (int64_t u = 0; u < int64_t(n); u++)
                for (size_t k = is[u]; k < is[u + 1]; k++) ITp[k] = uint32_t(u);
            stable_scatter(m, n, nt, [IFp](size_t k) { return IFp[k]; },
                           [ITp, IWp, OTp, OWp](size_t k, size_t p) {
                               OTp[p] = ITp[k];
                               OWp[p] = IWp[k];
                           },
                           os);
        }
    }

    auto& off = g->off;
    off.resize(n + 1);
    for (size_t v = 0; v <= n; v++) off[v] = os[v] + is[v];
    g->nb.resize(2 * m);
    g->w.resize(2 * m);
    g->strength_in.assign(n, 0.0);
    g->strength_out.assign(n, 0.0);
    g->self_weight.assign(n, 0.0);
    uint32_t* NB = g->nb.data();
    double* WW = g->w.data();
    double* SI = g->strength_in.data();
    double* SO = g->strength_out.data();
    double* SW = g->self_weight.data();
    const uint32_t* OT = out_to.data();
    const double* OWv = out_w.data();
    const uint32_t* IF = in_from.data();
    const double* IWv = in_w.data();
#pragma omp parallel for schedule(dynamic, 256) num_threads(nt)
    for (int64_t vv = 0; vv < int64_t(n); vv++) {
        const size_t v = size_t(vv);
        size_t idx = off[v];
        size_t i1 = os[v], j1 = os[v + 1], i2 = is[v], j2 = is[v + 1];
        while (i1 < j1 && i2 < j2) {
            uint32_t n1 = OT[i1], n2 = IF[i2];
            if (n1 < n2) {
                NB[idx] = n1; WW[idx++] = OWv[i1++];
            } else if (n1 > n2) {
                NB[idx] = n2 | IN_FLAG; WW[idx++] = IWv[i2++];
            } else {
                NB[idx] = n1; WW[idx++] = OWv[i1++];
                NB[idx] = n2 | IN_FLAG; WW[idx++] = IWv[i2++];
            }
        }
        while (i1 < j1) {
            NB[idx] = OT[i1]; WW[idx++] = OWv[i1++];
        }
        while (i2 < j2) {
            NB[idx] = IF[i2] | IN_FLAG; WW[idx++] = IWv[i2++];
        }
        // Graph::init_admin adds w to strength_out[from] / strength_in[to]
        // in increasing edge id: per vertex that is the contiguous id range
        // of its out-edges and the in-list (ids increasing).
        double so = 0.0, si = 0.0;
        int64_t self_e = -1;
        for (size_t e = os[v]; e < os[v + 1]; e++) {
            so += W[e];
            if (T[e] == v) self_e = int64_t(e);
        }
        for (size_t k = is[v]; k < is[v + 1]; k++) si += IWv[k];
        SO[v] = so;
        SI[v] = si;
        if (self_e >= 0) SW[v] = W[self_e];
    }
    g->out_off = std::move(os);
    g->out_nb = std::move(out_to);
    g->out_w = std::move(out_w);
    g->total_weight = 0.0;
    for (size_t e = 0; e < m; e++) g->total_weight += W[e];
    g->total_size = 0;
    for (size_t v = 0; v < n; v++) g->total_size += g->node_size[v];
    return g.release();
}

// ---------------------------------------------------------------------------
// MutableVertexPartition + RBConfigurationVertexPartition
// ---------------------------------------------------------------------------
namespace {

class Partition {
public:
    const Graph* graph;
    double resolution;
    size_t n;
    std::vector<uint32_t> memb;

    size_t ncomm = 0;
    std::vector<double> csize_;
    std::vector<size_t> cnodes_;
    std::vector<double> wfrom_, wto_;  // _total_weight_from_comm / _to_comm
    std::vector<uint32_t> empty_;      // _empty_communities

    // Singleton partition.  init_admin() would add each edge weight to
    // wfrom[from] / wto[to] in edge id order (undirected: to both ends),
    // which is exactly how Graph::init_admin accumulates the strengths, so
    // the strengths are copied instead of recomputed.
    Partition(const Graph* g, double res) : graph(g), resolution(res), n(g->n) {
        memb.resize(n);
        for (size_t i = 0; i < n; i++) memb[i] = uint32_t(i);
        alloc();
        ncomm = n;
        for (size_t v = 0; v < n; v++) {
            csize_[v] = 0.0 + g->node_size[v];
            cnodes_[v] = 1;
            wfrom_[v] = g->strength_out_(v);
            wto_[v] = g->strength_in[v];
        }
        empty_.clear();
    }
    Partition(const Graph* g, double res, std::vector<uint32_t>&& membership)
        : graph(g), resolution(res), n(g->n), memb(std::move(membership)) {
        alloc();
        init_admin();
    }

    void alloc() {
        csize_.assign(n, 0.0);
        cnodes_.assign(n, 0);
        wfrom_.assign(n, 0.0);
        wto_.assign(n, 0.0);
    }

    inline double csize(size_t c) const { return c < ncomm ? csize_[c] : 0; }
    inline size_t cnodes(size_t c) const { return c < ncomm ? cnodes_[c] : 0; }
    inline double total_weight_from_comm(size_t c) const { return c < ncomm ? wfrom_[c] : 0.0; }
    inline double total_weight_to_comm(size_t c) const { return c < ncomm ? wto_[c] : 0.0; }

    void update_n_communities() {
        ncomm = 0;
        for (size_t i = 0; i < n; i++)
            if (memb[i] >= ncomm) ncomm = size_t(memb[i]) + 1;
    }

    // MutableVertexPartition::init_admin (_total_weight_in_comm and the
    // possible-edge counters are only used by quality()/CPM, not tracked).
    void init_admin() {
        update_n_communities();
        std::fill(csize_.begin(), csize_.begin() + ncomm, 0.0);
        std::fill(cnodes_.begin(), cnodes_.begin() + ncomm, 0);
        std::fill(wfrom_.begin(), wfrom_.begin() + ncomm, 0.0);
        std::fill(wto_.begin(), wto_.begin() + ncomm, 0.0);
        empty_.clear();
        for (size_t v = 0; v < n; v++) {
            uint32_t c = memb[v];
            csize_[c] += graph->node_size[v];
            cnodes_[c] += 1;
        }
        const size_t m = graph->m;
        const uint32_t* F = graph->efrom.data();
        const uint32_t* T = graph->eto.data();
        const double* W = graph->ew.data();
        if (graph->directed) {
            for (size_t e = 0; e < m; e++) {
                uint32_t vc = memb[F[e]], uc = memb[T[e]];
                double w = W[e];
                wfrom_[vc] += w;
                wto_[uc] += w;
            }
        } else {
            for (size_t e = 0; e < m; e++) {
                uint32_t vc = memb[F[e]], uc = memb[T[e]];
                double w = W[e];
                wfrom_[vc] += w;
                wto_[uc] += w;
                wfrom_[uc] += w;
                wto_[vc] += w;
            }
        }
        for (size_t c = 0; c < ncomm; c++)
            if (cnodes_[c] == 0) empty_.push_back(uint32_t(c));
    }

    size_t add_empty_community() {
        ncomm = ncomm + 1;
        if (ncomm > n)
            throw std::runtime_error(
                "There cannot be more communities than nodes, so there must already be an empty community.");
        size_t c = ncomm - 1;
        csize_[c] = 0;
        cnodes_[c] = 0;
        wfrom_[c] = 0;
        wto_[c] = 0;
        empty_.push_back(uint32_t(c));
        return c;
    }

    size_t get_empty_community() {
        if (empty_.empty()) add_empty_community();
        return empty_.back();
    }

    void move_node(size_t v, size_t new_comm) {
        if (new_comm >= ncomm) {
            if (new_comm < n) {
                while (new_comm >= ncomm) add_empty_community();
            } else {
                throw std::runtime_error("Cannot add new communities beyond the number of nodes.");
            }
        }
        double node_size = graph->node_size[v];
        size_t old_comm = memb[v];

        cnodes_[old_comm] -= 1;
        csize_[old_comm] -= node_size;
        if (cnodes_[old_comm] == 0) empty_.push_back(uint32_t(old_comm));

        if (cnodes_[new_comm] == 0) {
            auto it = empty_.rbegin();
            while (it != empty_.rend() && *it != new_comm) ++it;
            if (it != empty_.rend()) empty_.erase((++it).base());
        }
        cnodes_[new_comm] += 1;
        csize_[new_comm] += graph->node_size[v];

        // libleidenalg loops over the OUT list (updating from-weights) and then
        // the IN list (updating to-weights).  The two sequences touch disjoint
        // arrays, so a single pass over the flagged ALL list performs exactly
        // the same floating point operations in the same per-array order.
        const size_t b = graph->off[v], e = graph->off[v + 1];
        const uint32_t* nb = graph->nb.data();
        const double* ww = graph->w.data();
        double* wf = wfrom_.data();
        double* wt = wto_.data();
        if (graph->directed) {
            for (size_t i = b; i < e; i++) {
                double w = ww[i];
                if (nb[i] & Graph::IN_FLAG) {
                    wt[old_comm] -= w;
                    wt[new_comm] += w;
                } else {
                    wf[old_comm] -= w;
                    wf[new_comm] += w;
                }
            }
        } else {
            for (size_t i = b; i < e; i++) {
                double w = ww[i];
                wf[old_comm] -= w;
                wf[new_comm] += w;
                wt[old_comm] -= w;
                wt[new_comm] += w;
            }
        }
        memb[v] = uint32_t(new_comm);
    }

    // rank_order_communities + relabel_communities
    void renumber_communities() {
        const size_t nbc = ncomm;
        struct Row { size_t c, cs, cn; };
        // orderCSize: (size_t)csize desc, cnodes desc, index asc -- a strict
        // total order.  Rows with csize < 1 and no nodes compare equal on the
        // first two keys and sort after every other row, so they are placed
        // at the end in index order and only the others are sorted.
        std::vector<Row> rows, tail;
        rows.reserve(nbc);
        for (size_t i = 0; i < nbc; i++) {
            double cs = csize(i);
            Row r{i, size_t(cs), cnodes(i)};
            if (r.cs == 0 && r.cn == 0) tail.push_back(r); else rows.push_back(r);
        }
        std::sort(rows.begin(), rows.end(), [](const Row& A, const Row& B) {
            if (A.cs == B.cs) {
                if (A.cn == B.cn) return A.c < B.c;
                return A.cn > B.cn;
            }
            return A.cs > B.cs;
        });
        rows.insert(rows.end(), tail.begin(), tail.end());
        std::vector<uint32_t> new_id(nbc, 0);
        for (size_t i = 0; i < nbc; i++) new_id[rows[i].c] = uint32_t(i);
        relabel(new_id);
    }

    void relabel(const std::vector<uint32_t>& new_id) {
        const size_t old_nb = ncomm;
        for (size_t i = 0; i < n; i++) memb[i] = new_id[memb[i]];
        update_n_communities();
        const size_t nbc = ncomm;
        std::vector<double> ncs(nbc, 0), nwf(nbc, 0.0), nwt(nbc, 0.0);
        std::vector<size_t> ncn(nbc, 0);
        for (size_t c = 0; c < old_nb; c++) {
            size_t nc = new_id[c];
            if (cnodes_[c] > 0) {
                nwf[nc] = wfrom_[c];
                nwt[nc] = wto_[c];
                ncs[nc] = csize_[c];
                ncn[nc] = cnodes_[c];
            }
        }
        std::copy(ncs.begin(), ncs.end(), csize_.begin());
        std::copy(ncn.begin(), ncn.end(), cnodes_.begin());
        std::copy(nwf.begin(), nwf.end(), wfrom_.begin());
        std::copy(nwt.begin(), nwt.end(), wto_.begin());
        empty_.clear();
        for (size_t c = 0; c < nbc; c++)
            if (cnodes_[c] == 0) empty_.push_back(uint32_t(c));
    }

    void from_coarse_partition(const std::vector<uint32_t>& coarse_memb,
                               const std::vector<uint32_t>& coarse_node) {
        for (size_t v = 0; v < n; v++) memb[v] = coarse_memb[coarse_node[v]];
        init_admin();
    }
};

// Graph::collapse_graph(partition).  Each aggregated node's edge list
// depends only on its own members, so contiguous ranges of aggregated nodes
// are processed by different threads and concatenated in order.
Graph* collapse_graph(const Graph& g, const Partition& part, int nthreads) {
    const size_t K = part.ncomm;
    const size_t n = g.n;
    // get_communities(): members of each community in increasing node order
    std::vector<size_t> coff(K + 1, 0);
    for (size_t v = 0; v < n; v++) coff[part.memb[v] + 1]++;
    for (size_t c = 0; c < K; c++) coff[c + 1] += coff[c];
    std::vector<uint32_t> members(n);
    {
        std::vector<size_t> pos(coff.begin(), coff.end() - 1);
        for (size_t v = 0; v < n; v++) members[pos[part.memb[v]]++] = uint32_t(v);
    }

    const int nt = (nthreads > 1 && g.m > 50000) ? nthreads : 1;
    // split [0, K) into nt ranges of roughly equal adjacency volume
    std::vector<size_t> bounds(nt + 1, K);
    bounds[0] = 0;
    if (nt > 1) {
        const double total = double(g.off[n]);
        size_t vc = 0;
        double acc = 0.0;
        for (int t = 1; t < nt; t++) {
            const double target = total * t / nt;
            while (vc < K && acc < target) {
                for (size_t k = coff[vc]; k < coff[vc + 1]; k++) {
                    const uint32_t v = members[k];
                    acc += double(g.off[v + 1] - g.off[v]);
                }
                vc++;
            }
            bounds[t] = vc;
        }
    }

    struct Chunk { std::vector<uint32_t> a, b; std::vector<double> w; };
    std::vector<Chunk> chunks(nt);
    const uint32_t* nb = g.nb.data();
    const double* ww = g.w.data();
    const uint32_t* memb = part.memb.data();
#pragma omp parallel num_threads(nt)
    {
        const int t = nt > 1 ? omp_get_thread_num() : 0;
        Chunk& ch = chunks[t];
        std::vector<double> ew_to_comm(K, 0.0);
        std::vector<char> added(K, 0);
        std::vector<uint32_t> neigh;
        const size_t kend = coff[bounds[t + 1]];
        for (size_t vc = bounds[t]; vc < bounds[t + 1]; vc++) {
            neigh.clear();
            for (size_t k = coff[vc]; k < coff[vc + 1]; k++) {
                const uint32_t v = members[k];
                // prefetch (performance hint only) the lists of later members
                if (k + 16 < kend) {
                    const uint32_t v2 = members[k + 16];
                    __builtin_prefetch(g.directed ? &g.out_off[v2] : &g.off[v2]);
                }
                if (k + 8 < kend) {
                    const uint32_t v1 = members[k + 8];
                    if (g.directed) {
                        const size_t b1 = g.out_off[v1];
                        __builtin_prefetch(&g.out_nb[b1]);
                        __builtin_prefetch(&g.out_w[b1]);
                        __builtin_prefetch(&g.out_w[b1] + 8);
                        __builtin_prefetch(&g.out_w[b1] + 16);
                    } else {
                        const size_t b1 = g.off[v1];
                        __builtin_prefetch(&g.nb[b1]);
                        __builtin_prefetch(&g.w[b1]);
                        __builtin_prefetch(&g.w[b1] + 8);
                    }
                }
                const size_t b = g.off[v], e = g.off[v + 1];
                if (g.directed) {
                    // get_neighbour_edges(v, OUT)
                    const uint32_t* onb = g.out_nb.data();
                    const double* ow = g.out_w.data();
                    for (size_t i = g.out_off[v], j = g.out_off[v + 1]; i < j; i++) {
                        uint32_t uc = memb[onb[i]];
                        if (!added[uc]) {
                            added[uc] = 1;
                            neigh.push_back(uc);
                        }
                        ew_to_comm[uc] += ow[i];
                    }
                } else {
                    // ALL list; "if (from != v) continue": keep the out part and
                    // the loops of the in part; loops are halved.
                    const size_t oc = g.outcnt[v];
                    for (size_t i = b; i < e; i++) {
                        uint32_t to = nb[i];
                        if (!(i - b < oc || to == v)) continue;
                        uint32_t uc = memb[to];
                        double w = ww[i];
                        if (to == v) w /= 2.0;
                        if (!added[uc]) {
                            added[uc] = 1;
                            neigh.push_back(uc);
                        }
                        ew_to_comm[uc] += w;
                    }
                }
            }
            for (uint32_t uc : neigh) {
                ch.a.push_back(uint32_t(vc));
                ch.b.push_back(uc);
                ch.w.push_back(ew_to_comm[uc]);
                ew_to_comm[uc] = 0.0;
                added[uc] = 0;
            }
        }
    }
    std::vector<uint32_t> ea, eb;
    std::vector<double> cw;
    if (nt == 1) {
        ea = std::move(chunks[0].a);
        eb = std::move(chunks[0].b);
        cw = std::move(chunks[0].w);
    } else {
        size_t tot = 0;
        for (auto& ch : chunks) tot += ch.a.size();
        ea.reserve(tot);
        eb.reserve(tot);
        cw.reserve(tot);
        for (auto& ch : chunks) {
            ea.insert(ea.end(), ch.a.begin(), ch.a.end());
            eb.insert(eb.end(), ch.b.begin(), ch.b.end());
            cw.insert(cw.end(), ch.w.begin(), ch.w.end());
        }
    }
    std::vector<double> csizes(K, 0);
    for (size_t c = 0; c < K; c++) csizes[c] = part.csize(c);
    if (g.directed)
        return Graph::create_grouped_directed(K, std::move(ea), std::move(eb), std::move(cw),
                                              std::move(csizes), nthreads);
    return Graph::create(K, false, ea, eb, std::move(cw), std::move(csizes));
}

// ---------------------------------------------------------------------------
// Optimiser (default settings: consider_comms=ALL_NEIGH_COMMS,
// refine_consider_comms=RAND_NEIGH_COMM, optimise_routine=MOVE_NODES,
// refine_routine=MERGE_NODES, refine_partition=true,
// consider_empty_community=true, max_comm_size=min_comm_size=0, so
// improvement_community_constraints() is identically 0.0).
//
// libleidenalg caches, per partition and per mode, the weight from the node
// being processed to each community (cache_neigh_communities), keyed by the
// node id.  Within move_nodes / merge_nodes_constrained the keys are invalid
// at the start (relabel_communities / init_admin reset them) and every visit
// of a node is preceded by a visit of another node that recomputed the
// caches, so the cached values are always those computed from the
// memberships at the time of the visit.  They are computed here directly
// from the flagged ALL list, accumulating every community's weights over
// the same entries in the same order.
// ---------------------------------------------------------------------------
static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

class Optimiser {
public:
    RNG rng;
    int nthreads = 1;
    bool debug = std::getenv("AOF_LEIDEN_DEBUG") != nullptr;

    // Optimiser::move_nodes(..., ALL_NEIGH_COMMS, consider_empty_community=true)
    double move_nodes(Partition& P) {
        const Graph& g = *P.graph;
        const size_t n = g.n;
        const bool directed = g.directed;
        const double res = P.resolution;
        double total_improv = 0.0;

        std::vector<uint32_t> q(n);
        for (size_t v = 0; v < n; v++) q[v] = uint32_t(v);
        shuffle(q, rng);
        // Ring-buffer deque; a node is never queued twice, so n slots suffice.
        size_t qhead = 0, qsize = n;
        std::vector<char> is_stable(n, 0);
        std::vector<char> comm_added(n + 1, 0);
        std::vector<uint32_t> comms;
        std::vector<double> cw_out(n + 1, 0.0), cw_in(n + 1, 0.0), cw_all;
        std::vector<uint32_t> cl_all;
        if (!g.all_positive) cw_all.assign(n + 1, 0.0);
        double* CWO = cw_out.data();
        double* CWI = directed ? cw_in.data() : cw_out.data();

        const uint32_t* NB = g.nb.data();
        const double* WW = g.w.data();
        uint32_t* M = P.memb.data();
        const double tw = g.total_weight * (2.0 - int(directed));

        const size_t* OFF = g.off.data();
        while (qsize > 0) {
            const size_t v = q[qhead];
            // prefetch (performance hint only) for the nodes queued next
            if (qsize > 16) {
                size_t p2 = qhead + 16; if (p2 >= n) p2 -= n;
                const uint32_t v2 = q[p2];
                __builtin_prefetch(&OFF[v2]);
                __builtin_prefetch(&M[v2]);
            }
            if (qsize > 8) {
                size_t p1 = qhead + 8; if (p1 >= n) p1 -= n;
                const uint32_t v1 = q[p1];
                const size_t b1 = OFF[v1];
                __builtin_prefetch(&NB[b1]);
                __builtin_prefetch(&NB[b1] + 16);
                __builtin_prefetch(&WW[b1]);
                __builtin_prefetch(&WW[b1] + 8);
                __builtin_prefetch(&WW[b1] + 16);
                __builtin_prefetch(&WW[b1] + 24);
                __builtin_prefetch(&P.cnodes_[M[v1]]);
                __builtin_prefetch(&g.strength_in[v1]);
                __builtin_prefetch(&g.self_weight[v1]);
                if (directed) __builtin_prefetch(&g.strength_out[v1]);
            }
            qhead = (qhead + 1 == n) ? 0 : qhead + 1;
            qsize--;

            const size_t v_comm = M[v];
            const size_t b = g.off[v], e = g.off[v + 1];

            // get_neigh_comms(v, ALL) (+ dedup via comm_added) and the OUT/IN
            // weight caches used by diff_move.
            if (g.all_positive) {
                // cached ALL weight is > 0 after every addition, so every
                // entry's community is pushed: first-occurrence order.
                for (size_t i = b; i < e; i++) {
                    const uint32_t raw = NB[i];
                    const uint32_t u = raw & Graph::ID_MASK;
                    const uint32_t comm = M[u];
                    double w = WW[i];
                    if (!comm_added[comm]) {
                        comm_added[comm] = 1;
                        comms.push_back(comm);
                    }
                    if (directed) {
                        if (raw & Graph::IN_FLAG) CWI[comm] += w; else CWO[comm] += w;
                    } else {
                        if (u == v) w /= 2.0;
                        CWO[comm] += w;
                    }
                }
            } else {
                cl_all.clear();
                for (size_t i = b; i < e; i++) {
                    const uint32_t raw = NB[i];
                    const uint32_t u = raw & Graph::ID_MASK;
                    const uint32_t comm = M[u];
                    double w = WW[i];
                    if (u == v && !directed) w /= 2.0;
                    cw_all[comm] += w;
                    if (cw_all[comm] != 0) cl_all.push_back(comm);
                    if (directed) {
                        if (raw & Graph::IN_FLAG) CWI[comm] += w; else CWO[comm] += w;
                    } else {
                        CWO[comm] += w;
                    }
                }
                for (uint32_t comm : cl_all) {
                    if (!comm_added[comm]) {
                        comm_added[comm] = 1;
                        comms.push_back(comm);
                    }
                }
            }
            const size_t n_neigh = comms.size();
            if (P.cnodes(v_comm) > 1) {
                size_t comm = P.get_empty_community();
                comms.push_back(uint32_t(comm));
                comm_added[comm] = 1;
            }

            size_t max_comm = v_comm;
            double max_improv = 10 * DBL_EPSILON;
            // RBConfigurationVertexPartition::diff_move; the "old" half is the
            // same for every candidate and is evaluated once.
            const double k_out = g.strength_out_(v);
            const double k_in = g.strength_in[v];
            const double self_weight = g.self_weight[v];
            double diff_old = 0.0;
            if (tw != 0.0) {
                double w_to_old = v_comm < P.ncomm ? CWO[v_comm] : 0.0;
                double w_from_old = v_comm < P.ncomm ? CWI[v_comm] : 0.0;
                double K_out_old = P.total_weight_from_comm(v_comm);
                double K_in_old = P.total_weight_to_comm(v_comm);
                diff_old = (w_to_old - res * k_out * K_in_old / tw) +
                           (w_from_old - res * k_in * K_out_old / tw);
            }
            for (uint32_t comm : comms) {
                comm_added[comm] = 0;
                double diff = 0.0;
                if (tw != 0.0 && comm != v_comm) {
                    double w_to_new = comm < P.ncomm ? CWO[comm] : 0.0;
                    double w_from_new = comm < P.ncomm ? CWI[comm] : 0.0;
                    double K_out_new = P.total_weight_from_comm(comm) + k_out;
                    double K_in_new = P.total_weight_to_comm(comm) + k_in;
                    double diff_new = (w_to_new + self_weight - res * k_out * K_in_new / tw) +
                                      (w_from_new + self_weight - res * k_in * K_out_new / tw);
                    diff = diff_new - diff_old;
                }
                double possible_improv = 0.0;
                if (comm != v_comm) possible_improv += 0.0;
                possible_improv += 1.0 * diff;
                if (possible_improv > max_improv) {
                    max_comm = comm;
                    max_improv = possible_improv;
                }
            }
            // reset the weight caches
            if (g.all_positive) {
                for (size_t k = 0; k < n_neigh; k++) {
                    uint32_t c = comms[k];
                    CWO[c] = 0;
                    CWI[c] = 0;
                }
            } else {
                for (size_t i = b; i < e; i++) {
                    uint32_t c = M[NB[i] & Graph::ID_MASK];
                    CWO[c] = 0;
                    CWI[c] = 0;
                    cw_all[c] = 0;
                }
            }
            comms.clear();
            is_stable[v] = 1;

            if (max_comm != v_comm) {
                total_improv += max_improv;
                P.move_node(v, max_comm);
                for (size_t i = b; i < e; i++) {
                    const uint32_t u = NB[i] & Graph::ID_MASK;
                    if (is_stable[u] && M[u] != max_comm) {
                        size_t tail = qhead + qsize;
                        if (tail >= n) tail -= n;
                        q[tail] = u;
                        qsize++;
                        is_stable[u] = 0;
                    }
                }
            }
        }
        P.renumber_communities();
        return total_improv;
    }

    // Optimiser::merge_nodes_constrained(..., RAND_NEIGH_COMM, constrained, false)
    double merge_nodes_constrained(Partition& P, const Partition& C) {
        const Graph& g = *P.graph;
        const size_t n = g.n;
        const bool directed = g.directed;
        const double res = P.resolution;
        double total_improv = 0.0;
        std::vector<uint32_t> order(n);
        for (size_t v = 0; v < n; v++) order[v] = uint32_t(v);
        shuffle(order, rng);

        const uint32_t* NB = g.nb.data();
        const double* WW = g.w.data();
        const uint32_t* CM = C.memb.data();
        uint32_t* M = P.memb.data();
        const double tw = g.total_weight * (2.0 - int(directed));
        // get_neigh_comms(v, ALL, constrained) (with duplicates), and the list
        // indices of those entries.  Sub-communities never straddle
        // constrained communities and v is alone in its sub-community, so
        // every entry of v's old and new sub-community is among them.
        size_t maxdeg = 0;
        for (size_t v = 0; v < n; v++) maxdeg = std::max(maxdeg, g.off[v + 1] - g.off[v]);
        std::vector<uint32_t> neigh_buf(maxdeg + 1), nidx_buf(maxdeg + 1);
        uint32_t* neigh = neigh_buf.data();
        uint32_t* nidx = nidx_buf.data();
        const size_t* OFF = g.off.data();
        for (size_t oi_ = 0; oi_ < n; oi_++) {
            const uint32_t v = order[oi_];
            // prefetch (performance hint only): node data 16 ahead, adjacency 8 ahead
            if (oi_ + 16 < n) {
                const uint32_t v2 = order[oi_ + 16];
                __builtin_prefetch(&OFF[v2]);
                __builtin_prefetch(&M[v2]);
                __builtin_prefetch(&CM[v2]);
            }
            if (oi_ + 8 < n) {
                const uint32_t v1 = order[oi_ + 8];
                const size_t b1 = OFF[v1];
                __builtin_prefetch(&NB[b1]);
                __builtin_prefetch(&NB[b1] + 16);
                __builtin_prefetch(&WW[b1]);
                __builtin_prefetch(&WW[b1] + 8);
                __builtin_prefetch(&WW[b1] + 16);
                __builtin_prefetch(&WW[b1] + 24);
                __builtin_prefetch(&P.cnodes_[M[v1]]);
            }
            const size_t v_comm = M[v];
            if (P.cnodes(v_comm) != 1) continue;
            const size_t b = g.off[v], e = g.off[v + 1];
            const uint32_t cv = CM[v];
            size_t k = 0;
            for (size_t i = b; i < e; i++) {
                uint32_t u = NB[i] & Graph::ID_MASK;
                if (cv == CM[u]) {
                    neigh[k] = M[u];
                    nidx[k++] = uint32_t(i - b);
                }
            }
            if (k == 0) continue;
            if (!(rng.integer(0, k) > 0)) continue;
            const size_t comm = neigh[rng.integer(0, k - 1)];

            double diff = 0.0;
            if (tw != 0.0 && comm != v_comm) {
                // weight_to/from_comm for the old and new community: sums over
                // the OUT (IN) entries of those communities, in list order.
                double to_old = 0.0, to_new = 0.0, from_old = 0.0, from_new = 0.0;
                for (size_t q = 0; q < k; q++) {
                    const uint32_t cu = neigh[q];
                    if (cu != v_comm && cu != comm) continue;
                    const size_t i = b + nidx[q];
                    const uint32_t raw = NB[i];
                    const uint32_t u = raw & Graph::ID_MASK;
                    double w = WW[i];
                    if (directed) {
                        if (raw & Graph::IN_FLAG) {
                            if (cu == v_comm) from_old += w; else from_new += w;
                        } else {
                            if (cu == v_comm) to_old += w; else to_new += w;
                        }
                    } else {
                        if (u == v) w /= 2.0;
                        if (cu == v_comm) to_old += w; else to_new += w;
                    }
                }
                if (!directed) {
                    from_old = to_old;
                    from_new = to_new;
                }
                const double k_out = g.strength_out_(v);
                const double k_in = g.strength_in[v];
                const double self_weight = g.self_weight[v];
                const double K_out_old = P.total_weight_from_comm(v_comm);
                const double K_in_old = P.total_weight_to_comm(v_comm);
                const double K_out_new = P.total_weight_from_comm(comm) + k_out;
                const double K_in_new = P.total_weight_to_comm(comm) + k_in;
                const double diff_old = (to_old - res * k_out * K_in_old / tw) +
                                        (from_old - res * k_in * K_out_old / tw);
                const double diff_new = (to_new + self_weight - res * k_out * K_in_new / tw) +
                                        (from_new + self_weight - res * k_in * K_out_new / tw);
                diff = diff_new - diff_old;
            }
            size_t max_comm = v_comm;
            double max_improv = 0;
            double possible_improv = 0.0;
            if (comm != v_comm) possible_improv += 0.0;
            possible_improv += 1.0 * diff;
            if (possible_improv >= max_improv) {
                max_comm = comm;
                max_improv = possible_improv;
            }
            if (max_comm != v_comm) {
                total_improv += max_improv;
                P.move_node(v, max_comm);
            }
        }
        P.renumber_communities();
        return total_improv;
    }

    // Optimiser::optimise_partition (single layer, no fixed nodes)
    double optimise_partition(Partition& P) {
        const Graph* G = P.graph;
        const size_t n = G->n;
        std::unique_ptr<Graph> owned_graph;     // collapsed graph if != G
        std::unique_ptr<Partition> owned_part;  // collapsed partition if != &P
        const Graph* cgraph = G;
        Partition* cpart = &P;

        std::vector<uint32_t> agg(n);
        for (size_t v = 0; v < n; v++) agg[v] = uint32_t(v);
        // partitions[0]->from_coarse_partition(collapsed, aggregate_node) is
        // called on every level > 1; each call recomputes the whole state of P
        // from (collapsed membership, aggregate map) and P is not read again on
        // those levels, so only the last call is performed (after the loop).
        bool coarse_pending = false;
        std::vector<uint32_t> coarse_memb, coarse_agg;
        bool aggregate_further = true;
        double improv = 0.0;
        int level = 0;
        do {
            double t0 = debug ? now_s() : 0;
            improv += move_nodes(*cpart);
            double t1 = debug ? now_s() : 0;

            if (cpart != &P) {
                coarse_memb = cpart->memb;
                coarse_agg = agg;
                coarse_pending = true;
            }

            Partition sub(cgraph, cpart->resolution);
            merge_nodes_constrained(sub, *cpart);
            double t2 = debug ? now_s() : 0;

            for (size_t v = 0; v < n; v++) agg[v] = sub.memb[agg[v]];

            std::unique_ptr<Graph> new_graph(collapse_graph(*cgraph, sub, nthreads));
            std::vector<uint32_t> new_memb(new_graph->n, 0);
            for (size_t v = 0; v < cgraph->n; v++) new_memb[sub.memb[v]] = cpart->memb[v];
            std::unique_ptr<Partition> new_part(
                new Partition(new_graph.get(), cpart->resolution, std::move(new_memb)));
            double t3 = debug ? now_s() : 0;
            if (debug)
                std::fprintf(stderr, "  level %d n=%zu m=%zu ncomm=%zu sub=%zu: move %.3fs merge %.3fs collapse %.3fs\n",
                             level, cgraph->n, cgraph->m, cpart->ncomm, sub.ncomm, t1 - t0, t2 - t1, t3 - t2);
            level++;

            aggregate_further = (new_graph->n < cgraph->n) && (cgraph->n > cpart->ncomm);

            owned_part = std::move(new_part);
            owned_graph = std::move(new_graph);
            cpart = owned_part.get();
            cgraph = owned_graph.get();
        } while (aggregate_further);

        owned_part.reset();
        owned_graph.reset();
        if (coarse_pending) P.from_coarse_partition(coarse_memb, coarse_agg);
        P.renumber_communities();
        return improv;
    }
};

}  // namespace

std::vector<int64_t> find_partition(const Graph& g, double resolution, uint64_t seed,
                                    const Options& opt, double* quality_gain) {
    Partition P(&g, resolution);
    Optimiser optimiser;
    optimiser.nthreads = opt.n_threads;
    optimiser.rng.seed(seed);  // Optimiser::set_rng_seed

    // leidenalg Optimiser.optimise_partition (Python loop)
    int itr = 0;
    double diff = 0;
    const int n_iterations = opt.n_iterations;
    bool continue_iteration = itr < n_iterations || n_iterations < 0;
    while (continue_iteration) {
        double t0 = optimiser.debug ? now_s() : 0;
        double diff_inc = optimiser.optimise_partition(P);
        if (optimiser.debug)
            std::fprintf(stderr, "iter %d: diff_inc=%.17g %.3fs\n", itr, diff_inc, now_s() - t0);
        diff += diff_inc;
        itr++;
        if (n_iterations < 0)
            continue_iteration = (diff_inc > 0);
        else
            continue_iteration = itr < n_iterations;
    }
    if (quality_gain) *quality_gain = diff;
    return std::vector<int64_t>(P.memb.begin(), P.memb.end());
}

}  // namespace aof_leiden
