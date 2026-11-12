#include "subset.hpp"

#include <omp.h>

#include <atomic>
#include <algorithm>
#include <cmath>

namespace aof {

template <class I>
void csr_subset_count(const int64_t* indptr, const I* indices, const int64_t* rows,
                      int64_t n_rows_out, const int32_t* col_map, int64_t* out_indptr) {
    out_indptr[0] = 0;
#pragma omp parallel for schedule(dynamic, 1024)
    for (int64_t r = 0; r < n_rows_out; ++r) {
        const int64_t src = rows ? rows[r] : r;
        const int64_t a = indptr[src], b = indptr[src + 1];
        int64_t k = b - a;
        if (col_map) {
            k = 0;
            for (int64_t e = a; e < b; ++e) k += col_map[indices[e]] >= 0;
        }
        out_indptr[r + 1] = k;
    }
    for (int64_t r = 0; r < n_rows_out; ++r) out_indptr[r + 1] += out_indptr[r];
}

template <class I, class V, class J>
void csr_subset_fill(const int64_t* indptr, const I* indices, const V* data, const int64_t* rows,
                     int64_t n_rows_out, const int32_t* col_map, const int64_t* out_indptr,
                     J* out_indices, V* out_data) {
#pragma omp parallel for schedule(dynamic, 1024)
    for (int64_t r = 0; r < n_rows_out; ++r) {
        const int64_t src = rows ? rows[r] : r;
        const int64_t a = indptr[src], b = indptr[src + 1];
        int64_t o = out_indptr[r];
        for (int64_t e = a; e < b; ++e) {
            const int64_t c = (int64_t)indices[e];
            const int64_t m = col_map ? col_map[c] : c;
            if (m < 0) continue;
            out_indices[o] = (J)m;
            out_data[o] = data[e];
            ++o;
        }
    }
}

#define AOF_SUBSET_I(I)                                                                       \
    template void csr_subset_count<I>(const int64_t*, const I*, const int64_t*, int64_t,      \
                                      const int32_t*, int64_t*);
AOF_SUBSET_I(int32_t)
AOF_SUBSET_I(int64_t)

#define AOF_SUBSET_IVJ(I, V, J)                                                               \
    template void csr_subset_fill<I, V, J>(const int64_t*, const I*, const V*, const int64_t*, \
                                           int64_t, const int32_t*, const int64_t*, J*, V*);
#define AOF_SUBSET_IV(I, V) AOF_SUBSET_IVJ(I, V, int32_t) AOF_SUBSET_IVJ(I, V, int64_t)
AOF_SUBSET_IV(int32_t, float)
AOF_SUBSET_IV(int32_t, double)
AOF_SUBSET_IV(int64_t, float)
AOF_SUBSET_IV(int64_t, double)


template <class V, class O>
void csr_scale_rows(const int64_t* indptr, int64_t n_rows, const V* data, const double* f,
                    bool divide, O* out) {
#pragma omp parallel for schedule(dynamic, 1024)
    for (int64_t r = 0; r < n_rows; ++r) {
        const double s = f[r];
        for (int64_t e = indptr[r]; e < indptr[r + 1]; ++e)
            out[e] = (O)(divide ? (double)data[e] / s : (double)data[e] * s);
    }
}

template <class V>
bool all_nonnegative_integers(const V* data, int64_t n) {
    std::atomic<bool> ok{true};
    const int64_t block = (int64_t)1 << 20;
    const int64_t nb = (n + block - 1) / block;
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t b = 0; b < nb; ++b) {
        if (!ok.load(std::memory_order_relaxed)) continue;
        const int64_t a = b * block, z = std::min(n, a + block);
        for (int64_t i = a; i < z; ++i) {
            const V v = data[i];
            if (!(v >= 0) || v != std::floor(v)) { ok.store(false, std::memory_order_relaxed); break; }
        }
    }
    return ok.load();
}

template void csr_scale_rows<float, float>(const int64_t*, int64_t, const float*, const double*, bool, float*);
template void csr_scale_rows<float, double>(const int64_t*, int64_t, const float*, const double*, bool, double*);
template void csr_scale_rows<double, double>(const int64_t*, int64_t, const double*, const double*, bool, double*);
template bool all_nonnegative_integers<float>(const float*, int64_t);
template bool all_nonnegative_integers<double>(const double*, int64_t);

}  // namespace aof
