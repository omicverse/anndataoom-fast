// Fused preprocessing passes over an on-disk CSR count matrix.
//
// Each pass decodes the matrix once on all threads.  The floating-point
// semantics mirror anndataoom/omicverse wherever they are visible in the
// output (float32 row sums, float64 size factors, float32 reciprocals and
// normalised values); internal accumulation is in double.
#pragma once

#include <cstdint>
#include <vector>

#include "engine.hpp"

namespace aof {

// ---- pass 1: QC metrics (+ per-gene counts over QC-passing cells) --------
enum QcStat : int { QC_TOTAL = 0, QC_NNZ = 1, QC_MT = 2, QC_RIBO = 3, QC_HB = 4 };
enum QcOp : int { OP_LT = 0, OP_LE = 1, OP_GT = 2, OP_GE = 3 };
struct QcRule { int stat; int op; double thr; };

struct QcResult {
    std::vector<double> total;               // per cell, float32-rounded sum
    std::vector<int64_t> nnz;                // stored entries per cell
    std::vector<double> mt, ribo, hb;        // f64(f32 group sum) / f64(f32 total)
    std::vector<uint8_t> keep;               // 1 if every rule passes
    std::vector<double> g_stored, g_nonzero, g_sum, g_sumsq;  // per gene, kept cells
};

// gene_group[c]: bit 0 = mitochondrial, bit 1 = ribosomal, bit 2 = haemoglobin,
// bit 7 = column not part of the matrix view (ignored entirely).
constexpr uint8_t kColDropped = 0x80;
// row_keep (optional, empty = all rows): rows outside the view are skipped.
QcResult qc_pass(const CSRSource& src, const std::vector<uint8_t>& gene_group,
                 const std::vector<QcRule>& rules, const std::vector<uint8_t>& row_keep);

// ---- pass 2: highly-expressed gene detection (normalize_total) ------------
struct HiResult {
    std::vector<double> hi_count;  // per gene: #cells where x > max_fraction * rowsum
    std::vector<double> g_sum, g_sumsq;  // per gene raw sum / sum of squares
    std::vector<double> rowsum;    // per cell sum over kept genes
};
HiResult hi_pass(const CSRSource& src, const std::vector<uint8_t>& row_keep,
                 const std::vector<uint8_t>& col_keep, double max_fraction);

// ---- pass 3: size factors + scale statistics + Pearson residual sums -------
struct StatsParams {
    double target_sum = 5e5;
    bool log1p = true;
    bool do_scale = true;
    bool do_pearson = true;
    double theta = 100.0;
    double clip = 0.0;            // residual clip (|r| <= clip); <= 0 disables
    double sum_total = 0.0;       // sum of sums_genes
    // Optional per-row overrides (size n_rows, or empty):
    std::vector<double> cells_override;  // per-cell totals used for the Pearson mu
    std::vector<float> recip_override;   // per-cell float32 reciprocal size factor
};
struct StatsResult {
    std::vector<float> cps;       // per cell: counts used for the size factor
    std::vector<double> nf;       // size factor cps / target_sum (0/nan/inf -> 1)
    std::vector<float> recip;     // float32 1 / float32(nf), as applied by anndataoom
    std::vector<double> g_sum, g_sumsq;  // per kept gene: raw sum / sum of squares
    std::vector<double> s1, s2;   // per kept gene: sum v, sum v^2 of normalised values
    std::vector<double> d1, d2;   // per kept gene: sum (r - r0), sum (r^2 - r0^2) over nonzeros
};
// col_map[c] = compact index of gene c among kept genes, or -1.
// excl[c] = 1 if gene c is excluded from the size-factor counts.
StatsResult stats_pass(const CSRSource& src, const std::vector<uint8_t>& row_keep,
                       const std::vector<int32_t>& col_map, const std::vector<uint8_t>& excl,
                       const std::vector<double>& sums_genes, const StatsParams& p);

// Contribution of all-zero entries to the Pearson residual sums, grouped by
// distinct per-cell count totals: sum_u m_u r0(s_u, g) and sum_u m_u r0^2.
void pearson_zero_terms(const std::vector<double>& s_unique, const std::vector<double>& mult,
                        const std::vector<double>& sums_genes, double sum_total, double theta,
                        double clip, std::vector<double>& z1, std::vector<double>& z2);

// ---- pass 4: extract the scaled HVG matrix --------------------------------
struct ScaledCSR {
    int64_t n_rows = 0, n_cols = 0;
    std::vector<int64_t> indptr;
    std::vector<int32_t> indices;
    std::vector<float> data;      // scaled + clipped value of each stored entry
    std::vector<float> z;         // scaled + clipped value of a zero, per column
};
ScaledCSR extract_scaled(const CSRSource& src, const std::vector<uint8_t>& row_keep,
                         const std::vector<int32_t>& col_map, const std::vector<double>& mean,
                         const std::vector<double>& std, const std::vector<float>& recip,
                         bool log1p, double clip);

// Dense-equivalent operator A = 1 z^T + (S - z on the stored pattern).  Both
// products stream the CSR rows; A^T @ P reduces per fixed row block in order.
class ScaledOperator {
public:
    explicit ScaledOperator(ScaledCSR m) : m_(std::move(m)) {}
    int64_t n_rows() const { return m_.n_rows; }
    int64_t n_cols() const { return m_.n_cols; }
    int64_t nnz() const { return (int64_t)m_.data.size(); }
    // Y[n_rows x k] = A @ Q[n_cols x k]   (row-major double)
    void matmul(const double* Q, int64_t k, double* Y) const;
    // Z[n_cols x k] = A^T @ P[n_rows x k]
    void rmatmul(const double* P, int64_t k, double* Z) const;
    const ScaledCSR& csr() const { return m_; }

private:
    ScaledCSR m_;
};

// ---- small deterministic dense kernels --------------------------------------
// In-place orthonormalisation of the columns of Y (n x k, row-major) by
// CholeskyQR2 (shifted CholeskyQR3 if Y is ill-conditioned).  The Gram matrix
// is reduced over fixed row blocks in order, so the result is independent of
// the number of threads.
void orthonormalize(double* Y, int64_t n, int64_t k);
// C[n x m] = A[n x k] @ B[k x m], all row-major; row-parallel, deterministic.
void gemm_tall(const double* A, const double* B, int64_t n, int64_t k, int64_t m, double* C);

}  // namespace aof
