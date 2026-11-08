#include "kernels.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace aof {

namespace {

inline bool rule_ok(double v, int op, double thr) {
    switch (op) {
        case OP_LT: return v < thr;
        case OP_LE: return v <= thr;
        case OP_GT: return v > thr;
        case OP_GE: return v >= thr;
    }
    return false;
}

inline double clip_to(double r, double c) {
    if (c > 0) {
        if (r > c) return c;
        if (r < -c) return -c;
    }
    return r;
}

// float32 log1p evaluated in double and rounded once: (almost always)
// correctly rounded and identical on every platform.  numpy's float32 log1p
// is neither (SVML on AVX-512 machines, libm elsewhere; up to 2 ulp apart).
inline float log1p_f32(float v) { return (float)std::log1p((double)v); }

void check_len(size_t got, int64_t want, const char* what) {
    if ((int64_t)got != want)
        throw std::invalid_argument(std::string(what) + " has the wrong length");
}

// Call body(float{}) or body(double{}) depending on the on-disk value dtype.
template <class Body>
auto dispatch_values(const CSRSource& src, Body&& body) {
    if (src.float_values()) return body(float{});
    return body(double{});
}

}  // namespace

// ---------------------------------------------------------------------------
QcResult qc_pass(const CSRSource& src, const std::vector<uint8_t>& gene_group,
                 const std::vector<QcRule>& rules, const std::vector<uint8_t>& row_keep) {
    check_len(gene_group.size(), src.n_cols, "gene_group");
    if (!row_keep.empty()) check_len(row_keep.size(), src.n_rows, "row_keep");
    const bool all_rows = row_keep.empty();
    const int64_t n = src.n_rows, G = src.n_cols;
    QcResult r;
    r.total.assign(n, 0.0);
    r.nnz.assign(n, 0);
    r.mt.assign(n, 0.0);
    r.ribo.assign(n, 0.0);
    r.hb.assign(n, 0.0);
    r.keep.assign(n, 0);
    OrderedGeneSums acc(4, G, default_slots());

    dispatch_values(src, [&](auto tag) {
        using V = decltype(tag);
        for_each_block<V>(src, &acc.reducer(), [&](int64_t, int slot, const Block<V>& b) {
            double* g_st = acc.slot(slot, 0);
            double* g_nz = acc.slot(slot, 1);
            double* g_s = acc.slot(slot, 2);
            double* g_sq = acc.slot(slot, 3);
            for (int64_t row = b.row0; row < b.row1; ++row) {
                if (!all_rows && !row_keep[row]) continue;
                double tot = 0, mt = 0, ribo = 0, hb = 0;
                int64_t nnz = 0;
                const int64_t e0 = b.begin(row), e1 = b.end(row);
                for (int64_t e = e0; e < e1; ++e) {
                    const int32_t c = b.col[e];
                    if ((uint32_t)c >= (uint32_t)G) throw std::out_of_range("column index out of range");
                    const uint8_t grp = gene_group[c];
                    if (grp & kColDropped) continue;
                    const double v = (double)b.val[e];
                    tot += v;
                    ++nnz;
                    if (grp & 1) mt += v;
                    if (grp & 2) ribo += v;
                    if (grp & 4) hb += v;
                }
                // anndataoom: float32 row sums stored into float64, divided in float64.
                const double tot32 = (double)(float)tot;
                const double denom = tot32 == 0.0 ? 1.0 : tot32;
                const double f_mt = (double)(float)mt / denom, f_ribo = (double)(float)ribo / denom,
                             f_hb = (double)(float)hb / denom;
                r.total[row] = tot32;
                r.nnz[row] = nnz;
                r.mt[row] = f_mt;
                r.ribo[row] = f_ribo;
                r.hb[row] = f_hb;
                bool keep = true;
                for (const QcRule& q : rules) {
                    double stat;
                    switch (q.stat) {
                        case QC_TOTAL: stat = tot32; break;
                        case QC_NNZ: stat = (double)nnz; break;
                        case QC_MT: stat = f_mt; break;
                        case QC_RIBO: stat = f_ribo; break;
                        case QC_HB: stat = f_hb; break;
                        default: throw std::invalid_argument("unknown QC statistic");
                    }
                    if (!rule_ok(stat, q.op, q.thr)) { keep = false; break; }
                }
                r.keep[row] = keep ? 1 : 0;
                if (!keep) continue;
                for (int64_t e = e0; e < e1; ++e) {
                    const int32_t c = b.col[e];
                    if (gene_group[c] & kColDropped) continue;
                    const double v = (double)b.val[e];
                    g_st[c] += 1.0;
                    g_nz[c] += (v != 0.0);
                    g_s[c] += v;
                    g_sq[c] += v * v;
                }
            }
        });
        return 0;
    });

    r.g_stored = acc.take(0);
    r.g_nonzero = acc.take(1);
    r.g_sum = acc.take(2);
    r.g_sumsq = acc.take(3);
    return r;
}

// ---------------------------------------------------------------------------
HiResult hi_pass(const CSRSource& src, const std::vector<uint8_t>& row_keep,
                 const std::vector<uint8_t>& col_keep, double max_fraction) {
    check_len(row_keep.size(), src.n_rows, "row_keep");
    check_len(col_keep.size(), src.n_cols, "col_keep");
    const int64_t G = src.n_cols;
    HiResult r;
    r.rowsum.assign(src.n_rows, 0.0);
    OrderedGeneSums acc(3, G, default_slots());
    const float frac32 = (float)max_fraction;

    dispatch_values(src, [&](auto tag) {
        using V = decltype(tag);
        for_each_block<V>(src, &acc.reducer(), [&](int64_t, int slot, const Block<V>& b) {
            double* hi = acc.slot(slot, 0);
            double* gs = acc.slot(slot, 1);
            double* gq = acc.slot(slot, 2);
            for (int64_t row = b.row0; row < b.row1; ++row) {
                if (!row_keep[row]) continue;
                const int64_t e0 = b.begin(row), e1 = b.end(row);
                double s = 0;
                for (int64_t e = e0; e < e1; ++e)
                    if (col_keep[b.col[e]]) s += (double)b.val[e];
                r.rowsum[row] = s;
                const float thr = (float)s * frac32;  // float32, as numpy does
                for (int64_t e = e0; e < e1; ++e) {
                    const int32_t c = b.col[e];
                    if (!col_keep[c]) continue;
                    const double v = (double)b.val[e];
                    hi[c] += ((float)v > thr);
                    gs[c] += v;
                    gq[c] += v * v;
                }
            }
        });
        return 0;
    });
    r.hi_count = acc.take(0);
    r.g_sum = acc.take(1);
    r.g_sumsq = acc.take(2);
    return r;
}

// ---------------------------------------------------------------------------
StatsResult stats_pass(const CSRSource& src, const std::vector<uint8_t>& row_keep,
                       const std::vector<int32_t>& col_map, const std::vector<uint8_t>& excl,
                       const std::vector<double>& sums_genes, const StatsParams& p) {
    check_len(row_keep.size(), src.n_rows, "row_keep");
    check_len(col_map.size(), src.n_cols, "col_map");
    check_len(excl.size(), src.n_cols, "excl");
    const bool have_cells = !p.cells_override.empty(), have_recip = !p.recip_override.empty();
    if (have_cells) check_len(p.cells_override.size(), src.n_rows, "cells_override");
    if (have_recip) check_len(p.recip_override.size(), src.n_rows, "recip_override");
    int64_t K = 0;
    for (int32_t m : col_map) K = std::max<int64_t>(K, (int64_t)m + 1);
    if (p.do_pearson) check_len(sums_genes.size(), K, "sums_genes");

    StatsResult r;
    r.cps.assign(src.n_rows, 0.f);
    r.nf.assign(src.n_rows, 1.0);
    r.recip.assign(src.n_rows, 1.f);
    OrderedGeneSums acc(6, K, default_slots());

    dispatch_values(src, [&](auto tag) {
        using V = decltype(tag);
        for_each_block<V>(src, &acc.reducer(), [&](int64_t, int slot, const Block<V>& b) {
            double* s1 = acc.slot(slot, 0);
            double* s2 = acc.slot(slot, 1);
            double* d1 = acc.slot(slot, 2);
            double* d2 = acc.slot(slot, 3);
            double* gs = acc.slot(slot, 4);
            double* gq = acc.slot(slot, 5);
            for (int64_t row = b.row0; row < b.row1; ++row) {
                if (!row_keep[row]) continue;
                const int64_t e0 = b.begin(row), e1 = b.end(row);
                double s = 0;
                for (int64_t e = e0; e < e1; ++e) {
                    const int32_t c = b.col[e];
                    if (col_map[c] >= 0 && !excl[c]) s += (double)b.val[e];
                }
                // anndataoom: float32 row sum, float64 size factor, float32 reciprocal.
                const float cps32 = (float)s;
                double nf = (double)cps32 / p.target_sum;
                if (!std::isfinite(nf) || nf == 0.0) nf = 1.0;
                const float recip = have_recip ? p.recip_override[row] : 1.0f / (float)nf;
                r.cps[row] = cps32;
                r.nf[row] = nf;
                r.recip[row] = recip;
                const double cps = have_cells ? p.cells_override[row] : (double)cps32;
                for (int64_t e = e0; e < e1; ++e) {
                    const int32_t m = col_map[b.col[e]];
                    if (m < 0) continue;
                    const double x = (double)b.val[e];
                    gs[m] += x;
                    gq[m] += x * x;
                    if (p.do_scale) {
                        float v = (float)x * recip;
                        if (p.log1p) v = log1p_f32(v);
                        const double vd = (double)v;
                        s1[m] += vd;
                        s2[m] += vd * vd;
                    }
                    if (p.do_pearson) {
                        const double mu = cps * sums_genes[m] / p.sum_total;
                        double den = std::sqrt(mu + mu * mu / p.theta);
                        if (den == 0.0) den = 1.0;
                        const double rr = clip_to((x - mu) / den, p.clip);
                        const double r0 = clip_to((0.0 - mu) / den, p.clip);
                        d1[m] += rr - r0;
                        d2[m] += rr * rr - r0 * r0;
                    }
                }
            }
        });
        return 0;
    });

    r.s1 = acc.take(0);
    r.s2 = acc.take(1);
    r.d1 = acc.take(2);
    r.d2 = acc.take(3);
    r.g_sum = acc.take(4);
    r.g_sumsq = acc.take(5);
    return r;
}

void pearson_zero_terms(const std::vector<double>& s_unique, const std::vector<double>& mult,
                        const std::vector<double>& sums_genes, double sum_total, double theta,
                        double clip, std::vector<double>& z1, std::vector<double>& z2) {
    check_len(mult.size(), (int64_t)s_unique.size(), "mult");
    const int64_t G = (int64_t)sums_genes.size(), U = (int64_t)s_unique.size();
    z1.assign(G, 0.0);
    z2.assign(G, 0.0);
#pragma omp parallel for schedule(dynamic, 16)
    for (int64_t g = 0; g < G; ++g) {
        const double sg = sums_genes[g];
        double a1 = 0, a2 = 0;
        for (int64_t u = 0; u < U; ++u) {
            const double mu = s_unique[u] * sg / sum_total;
            double den = std::sqrt(mu + mu * mu / theta);
            if (den == 0.0) den = 1.0;
            const double r0 = clip_to((0.0 - mu) / den, clip);
            a1 += mult[u] * r0;
            a2 += mult[u] * (r0 * r0);
        }
        z1[g] = a1;
        z2[g] = a2;
    }
}

// ---------------------------------------------------------------------------
ScaledCSR extract_scaled(const CSRSource& src, const std::vector<uint8_t>& row_keep,
                         const std::vector<int32_t>& col_map, const std::vector<double>& mean,
                         const std::vector<double>& sd, const std::vector<float>& recip,
                         bool log1p, double clip) {
    check_len(row_keep.size(), src.n_rows, "row_keep");
    check_len(col_map.size(), src.n_cols, "col_map");
    check_len(recip.size(), src.n_rows, "recip");
    const int64_t K = (int64_t)mean.size();
    check_len(sd.size(), K, "std");

    int64_t n_out = 0;
    for (int64_t i = 0; i < src.n_rows; ++i) n_out += row_keep[i] != 0;

    struct Piece { std::vector<int64_t> rowcount; std::vector<int32_t> idx; std::vector<float> val; };
    std::vector<Piece> pieces(src.n_blocks());

    dispatch_values(src, [&](auto tag) {
        using V = decltype(tag);
        for_each_block<V>(src, nullptr, [&](int64_t bi, int, const Block<V>& b) {
            Piece& pc = pieces[bi];
            for (int64_t row = b.row0; row < b.row1; ++row) {
                if (!row_keep[row]) continue;
                const int64_t before = (int64_t)pc.idx.size();
                const float rc = recip[row];
                for (int64_t e = b.begin(row); e < b.end(row); ++e) {
                    const int32_t m = col_map[b.col[e]];
                    if (m < 0) continue;
                    float v = (float)b.val[e] * rc;
                    if (log1p) v = log1p_f32(v);
                    const double a = clip_to(((double)v - mean[m]) / sd[m], clip);
                    pc.idx.push_back(m);
                    pc.val.push_back((float)a);
                }
                pc.rowcount.push_back((int64_t)pc.idx.size() - before);
            }
        });
        return 0;
    });

    ScaledCSR out;
    out.n_rows = n_out;
    out.n_cols = K;
    out.indptr.assign(n_out + 1, 0);
    std::vector<int64_t> piece_off(pieces.size() + 1, 0);
    {
        int64_t row = 0;
        for (size_t i = 0; i < pieces.size(); ++i) {
            for (int64_t c : pieces[i].rowcount) {
                out.indptr[row + 1] = out.indptr[row] + c;
                ++row;
            }
            piece_off[i + 1] = piece_off[i] + (int64_t)pieces[i].idx.size();
        }
    }
    out.indices.resize(out.indptr[n_out]);
    out.data.resize(out.indptr[n_out]);
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t)pieces.size(); ++i) {
        Piece& pc = pieces[i];
        std::memcpy(out.indices.data() + piece_off[i], pc.idx.data(), pc.idx.size() * sizeof(int32_t));
        std::memcpy(out.data.data() + piece_off[i], pc.val.data(), pc.val.size() * sizeof(float));
        std::vector<int32_t>().swap(pc.idx);
        std::vector<float>().swap(pc.val);
    }
    out.z.resize(K);
    for (int64_t m = 0; m < K; ++m) out.z[m] = (float)clip_to((0.0 - mean[m]) / sd[m], clip);
    return out;
}

// ---------------------------------------------------------------------------
namespace {
constexpr int64_t kRowBlock = 4096;  // fixed reduction block (thread-independent)
constexpr int64_t kMaxK = 1024;

// Column sums of P (n x k) over fixed row blocks, merged in order.
std::vector<double> column_sums(const double* P, int64_t n, int64_t k) {
    const int64_t nb = (n + kRowBlock - 1) / kRowBlock;
    OrderedReducer red(k, default_slots());
    ordered_parallel(nb, &red, [&](int64_t b, int slot) {
        double* acc = red.slot(slot);
        const int64_t r1 = std::min(n, (b + 1) * kRowBlock);
        for (int64_t i = b * kRowBlock; i < r1; ++i) {
            const double* p = P + i * k;
            for (int64_t t = 0; t < k; ++t) acc[t] += p[t];
        }
    });
    return red.total();
}
}  // namespace

void ScaledOperator::matmul(const double* Q, int64_t k, double* Y) const {
    if (k > kMaxK) throw std::invalid_argument("k too large");
    const int64_t n = m_.n_rows, K = m_.n_cols;
    std::vector<double> zQ(k, 0.0);  // z^T Q in fixed order
    for (int64_t j = 0; j < K; ++j) {
        const double zj = m_.z[j];
        const double* q = Q + j * k;
        for (int64_t t = 0; t < k; ++t) zQ[t] += zj * q[t];
    }
#pragma omp parallel for schedule(dynamic, 256)
    for (int64_t i = 0; i < n; ++i) {
        double* y = Y + i * k;
        for (int64_t t = 0; t < k; ++t) y[t] = zQ[t];
        for (int64_t e = m_.indptr[i]; e < m_.indptr[i + 1]; ++e) {
            const int32_t j = m_.indices[e];
            const double w = (double)m_.data[e] - (double)m_.z[j];
            const double* q = Q + (int64_t)j * k;
            for (int64_t t = 0; t < k; ++t) y[t] += w * q[t];
        }
    }
}

void ScaledOperator::rmatmul(const double* P, int64_t k, double* Z) const {
    if (k > kMaxK) throw std::invalid_argument("k too large");
    const int64_t n = m_.n_rows, K = m_.n_cols;
    const std::vector<double> colsum = column_sums(P, n, k);
    // Z = z (x) colsum + sum_i sum_{j in row i} (a_ij - z_j) P[i, :], reduced per
    // fixed row block (one K x k partial per slot) and merged in block order.
    const int64_t nb = (n + kRowBlock - 1) / kRowBlock;
    OrderedReducer red(K * k, default_slots());
    ordered_parallel(nb, &red, [&](int64_t b, int slot) {
        double* acc = red.slot(slot);
        const int64_t r1 = std::min(n, (b + 1) * kRowBlock);
        for (int64_t i = b * kRowBlock; i < r1; ++i) {
            const double* p = P + i * k;
            for (int64_t e = m_.indptr[i]; e < m_.indptr[i + 1]; ++e) {
                const int32_t j = m_.indices[e];
                const double w = (double)m_.data[e] - (double)m_.z[j];
                double* zj = acc + (int64_t)j * k;
                for (int64_t t = 0; t < k; ++t) zj[t] += w * p[t];
            }
        }
    });
    const std::vector<double>& tot = red.total();
#pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < K; ++j) {
        const double zj = m_.z[j];
        for (int64_t t = 0; t < k; ++t) Z[j * k + t] = zj * colsum[t] + tot[j * k + t];
    }
}

// ---------------------------------------------------------------------------
namespace {

// Gram matrix G = Y^T Y (k x k, row-major) reduced over fixed row blocks.
std::vector<double> gram(const double* Y, int64_t n, int64_t k) {
    const int64_t nb = (n + kRowBlock - 1) / kRowBlock;
    OrderedReducer red(k * k, default_slots());
    ordered_parallel(nb, &red, [&](int64_t b, int slot) {
        double* g = red.slot(slot);
        const int64_t r1 = std::min(n, (b + 1) * kRowBlock);
        for (int64_t i = b * kRowBlock; i < r1; ++i) {
            const double* y = Y + i * k;
            for (int64_t a = 0; a < k; ++a) {
                const double ya = y[a];
                double* ga = g + a * k;
                for (int64_t c = a; c < k; ++c) ga[c] += ya * y[c];
            }
        }
    });
    std::vector<double> G = red.total();
    for (int64_t a = 0; a < k; ++a)
        for (int64_t c = 0; c < a; ++c) G[a * k + c] = G[c * k + a];
    return G;
}

// Upper Cholesky factor R (G = R^T R), in place in G; false if not PD.
bool cholesky_upper(std::vector<double>& G, int64_t k) {
    for (int64_t j = 0; j < k; ++j) {
        double d = G[j * k + j];
        for (int64_t p = 0; p < j; ++p) d -= G[p * k + j] * G[p * k + j];
        if (!(d > 0.0) || !std::isfinite(d)) return false;
        const double rjj = std::sqrt(d);
        G[j * k + j] = rjj;
        for (int64_t c = j + 1; c < k; ++c) {
            double s = G[j * k + c];
            for (int64_t p = 0; p < j; ++p) s -= G[p * k + j] * G[p * k + c];
            G[j * k + c] = s / rjj;
        }
    }
    for (int64_t a = 0; a < k; ++a)
        for (int64_t c = 0; c < a; ++c) G[a * k + c] = 0.0;
    return true;
}

// Y <- Y R^{-1}, row by row (forward substitution with the upper factor).
void apply_rinv(double* Y, int64_t n, int64_t k, const std::vector<double>& R) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        double* y = Y + i * k;
        for (int64_t c = 0; c < k; ++c) {
            double s = y[c];
            for (int64_t p = 0; p < c; ++p) s -= y[p] * R[p * k + c];
            y[c] = s / R[c * k + c];
        }
    }
}

bool cholqr_step(double* Y, int64_t n, int64_t k, double shift) {
    std::vector<double> G = gram(Y, n, k);
    for (int64_t a = 0; a < k; ++a) G[a * k + a] += shift;
    if (!cholesky_upper(G, k)) return false;
    apply_rinv(Y, n, k, G);
    return true;
}

}  // namespace

void orthonormalize(double* Y, int64_t n, int64_t k) {
    if (k == 0) return;
    if (cholqr_step(Y, n, k, 0.0)) {
        if (!cholqr_step(Y, n, k, 0.0))
            throw std::runtime_error("CholeskyQR2 failed on the second pass");
        return;
    }
    // Shifted CholeskyQR3 (Fukaya et al. 2020) for ill-conditioned inputs.
    std::vector<double> G = gram(Y, n, k);
    double fro2 = 0;
    for (int64_t a = 0; a < k; ++a) fro2 += G[a * k + a];
    const double u = std::numeric_limits<double>::epsilon() / 2;
    const double shift = 11.0 * ((double)n * k + (double)k * (k + 1)) * u * fro2;
    if (!cholqr_step(Y, n, k, shift) || !cholqr_step(Y, n, k, 0.0) || !cholqr_step(Y, n, k, 0.0))
        throw std::runtime_error("shifted CholeskyQR3 failed (rank-deficient block)");
}

void gemm_tall(const double* A, const double* B, int64_t n, int64_t k, int64_t m, double* C) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        const double* a = A + i * k;
        double* c = C + i * m;
        for (int64_t t = 0; t < m; ++t) c[t] = 0.0;
        for (int64_t p = 0; p < k; ++p) {
            const double ap = a[p];
            const double* b = B + p * m;
            for (int64_t t = 0; t < m; ++t) c[t] += ap * b[t];
        }
    }
}

}  // namespace aof
