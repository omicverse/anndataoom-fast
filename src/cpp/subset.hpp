#pragma once
#include <cstdint>

namespace aof {

// Row/column subset of a CSR matrix in two parallel passes.  `rows` (length
// n_rows_out, nullptr = all rows in order) picks and orders the output rows;
// `col_map` (length n_cols, nullptr = keep every column) gives each input
// column's output index, or -1 to drop it.  Stored entries keep their order,
// so sorted indices stay sorted.

// Pass 1: out_indptr (length n_rows_out + 1) from the kept entries per row.
template <class I>
void csr_subset_count(const int64_t* indptr, const I* indices, const int64_t* rows,
                      int64_t n_rows_out, const int32_t* col_map, int64_t* out_indptr);

// Pass 2: fill out_indices / out_data at the offsets from pass 1.
template <class I, class V, class J>
void csr_subset_fill(const int64_t* indptr, const I* indices, const V* data, const int64_t* rows,
                     int64_t n_rows_out, const int32_t* col_map, const int64_t* out_indptr,
                     J* out_indices, V* out_data);

}  // namespace aof

namespace aof {

// out[e] = data[e] / f[row] (or * f[row]) for every stored entry, computed in
// double (numpy's float32 / float64 -> float64 result, bit for bit).
template <class V, class O>
void csr_scale_rows(const int64_t* indptr, int64_t n_rows, const V* data, const double* f,
                    bool divide, O* out);

// True when every value is >= 0 and integral (stops early otherwise).
template <class V>
bool all_nonnegative_integers(const V* data, int64_t n);

}  // namespace aof
