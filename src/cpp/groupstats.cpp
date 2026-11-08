#include "groupstats.hpp"

#include <algorithm>
#include <cstring>

#include "engine.hpp"

namespace aof {

template <class V, class I>
GroupStats group_stats_csr(const int64_t* indptr, const I* indices, const V* data, int64_t G,
                           const int64_t* order, const int64_t* offsets, int64_t C) {
    GroupStats r;
    r.sum.assign((size_t)C * G, 0.0);
    r.sumsq.assign((size_t)C * G, 0.0);
    r.nnz.assign((size_t)C * G, 0.0);
    constexpr int64_t kChunk = 2048;  // rows per reduction chunk (thread-independent)
    for (int64_t c = 0; c < C; ++c) {
        const int64_t a = offsets[c], b = offsets[c + 1];
        if (a == b) continue;
        const int64_t nchunks = (b - a + kChunk - 1) / kChunk;
        OrderedReducer red(3 * G, default_slots());
        ordered_parallel(nchunks, &red, [&](int64_t ch, int slot) {
            double* s = red.slot(slot);
            double* q = s + G;
            double* z = s + 2 * G;
            const int64_t r0 = a + ch * kChunk, r1 = std::min(b, r0 + kChunk);
            for (int64_t rr = r0; r1 > rr; ++rr) {
                const int64_t row = order[rr];
                for (int64_t e = indptr[row]; e < indptr[row + 1]; ++e) {
                    const int64_t j = (int64_t)indices[e];
                    const double v = (double)data[e];
                    s[j] += v;
                    q[j] += v * v;
                    z[j] += 1.0;
                }
            }
        });
        const std::vector<double>& t = red.total();
        std::memcpy(&r.sum[c * G], t.data(), G * sizeof(double));
        std::memcpy(&r.sumsq[c * G], t.data() + G, G * sizeof(double));
        std::memcpy(&r.nnz[c * G], t.data() + 2 * G, G * sizeof(double));
    }
    return r;
}

#define AOF_INST(V, I)                                                                         \
    template GroupStats group_stats_csr<V, I>(const int64_t*, const I*, const V*, int64_t,    \
                                              const int64_t*, const int64_t*, int64_t);
AOF_INST(float, int32_t)
AOF_INST(float, int64_t)
AOF_INST(double, int32_t)
AOF_INST(double, int64_t)
#undef AOF_INST

}  // namespace aof
