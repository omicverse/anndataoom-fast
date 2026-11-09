// Streaming CSR engine: splits the rows of an on-disk CSR matrix into fixed
// blocks, decodes blocks on all threads, and reduces per-gene statistics in a
// thread-count-independent order.
//
// Determinism contract: block boundaries depend only on indptr and the block
// target size, never on the thread count.  Each block is processed by a
// single thread, in row order, into its own reduction slot; slots are merged
// into the running total strictly in block order.  Every floating-point
// reduction therefore happens in the same order on 1 thread or 64.
//
// Memory traffic is what limits these passes on many-core machines, so a
// block is decoded in cache-sized sub-blocks straight into typed buffers
// (float32 values whenever the on-disk dtype fits exactly), and threads never
// wait for each other except when a reduction slot is still in use.
#pragma once

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "reader.hpp"

namespace aof {

constexpr int64_t kSubBlockElems = 1 << 16;  // decode granularity (~512 KB of buffers)

struct CSRSource {
    std::shared_ptr<const Dataset> data, indices;
    std::vector<int64_t> indptr;  // n_rows + 1
    int64_t n_rows = 0, n_cols = 0;
    std::vector<int64_t> block_start;  // row boundaries, size n_blocks + 1

    CSRSource(std::shared_ptr<const Dataset> d, std::shared_ptr<const Dataset> ix,
              std::vector<int64_t> ip, int64_t ncols, int64_t target_block_elems) :
        data(std::move(d)), indices(std::move(ix)), indptr(std::move(ip)), n_cols(ncols) {
        if (indptr.empty()) throw std::invalid_argument("empty indptr");
        n_rows = (int64_t)indptr.size() - 1;
        if (indptr.front() != 0 || indptr.back() > data->size() ||
            indptr.back() > indices->size())
            throw std::invalid_argument("indptr inconsistent with data/indices");
        for (int64_t r = 0; r < n_rows; ++r)
            if (indptr[r + 1] < indptr[r]) throw std::invalid_argument("indptr not monotone");
        // Greedy row blocks of at least `target_block_elems` stored values.
        const int64_t max_rows = 1 << 16;
        block_start.push_back(0);
        int64_t r0 = 0;
        while (r0 < n_rows) {
            int64_t r1 = r0;
            while (r1 < n_rows && r1 - r0 < max_rows &&
                   indptr[r1] - indptr[r0] < target_block_elems)
                ++r1;
            if (r1 == r0) ++r1;
            block_start.push_back(r1);
            r0 = r1;
        }
    }
    int64_t n_blocks() const { return (int64_t)block_start.size() - 1; }
    // float32 holds every value of these dtypes exactly.
    bool float_values() const {
        switch (data->dtype()) {
            case DType::F32: case DType::I8: case DType::I16: case DType::U8: case DType::U16:
                return true;
            default:
                return false;
        }
    }
};

// A decoded run of consecutive rows.
template <class V>
struct Block {
    int64_t row0 = 0, row1 = 0, elem0 = 0;
    std::vector<V> val;
    std::vector<int32_t> col;
    const int64_t* indptr = nullptr;
    int64_t begin(int64_t r) const { return indptr[r] - elem0; }
    int64_t end(int64_t r) const { return indptr[r + 1] - elem0; }
};

// Separate decode scratch for data and indices, so each keeps its own
// last-decoded chunk (row runs usually start in the middle of a chunk).
template <class V>
struct ThreadCtx {
    ReadScratch rd_val, rd_col;
    Block<V> blk;
};

template <class V>
inline void load_rows(const CSRSource& src, int64_t r0, int64_t r1, ThreadCtx<V>& t) {
    Block<V>& k = t.blk;
    k.row0 = r0;
    k.row1 = r1;
    k.elem0 = src.indptr[r0];
    k.indptr = src.indptr.data();
    const int64_t e1 = src.indptr[r1];
    k.val.resize((size_t)(e1 - k.elem0));
    k.col.resize((size_t)(e1 - k.elem0));
    src.data->read_as<V>(k.elem0, e1, k.val.data(), t.rd_val);
    src.indices->read_as<int32_t>(k.elem0, e1, k.col.data(), t.rd_col);
}

// W reduction slots of `len` doubles each, merged into `total` in block order.
class OrderedReducer {
public:
    OrderedReducer(int64_t len, int n_slots) :
        len_(len), n_slots_(n_slots), slots_((size_t)n_slots * len, 0.0), total_(len, 0.0) {}
    int n_slots() const { return n_slots_; }
    int64_t len() const { return len_; }
    double* slot(int s) { return slots_.data() + (size_t)s * len_; }
    void merge(int s) {  // total += slot; slot = 0   (one thread at a time)
        double* p = slot(s);
        double* t = total_.data();
        for (int64_t i = 0; i < len_; ++i) {
            t[i] += p[i];
            p[i] = 0.0;
        }
    }
    const std::vector<double>& total() const { return total_; }

private:
    int64_t len_;
    int n_slots_;
    std::vector<double> slots_, total_;
};

// Per-gene statistics: `n_stats` arrays of `n_genes` in one reducer.
class OrderedGeneSums {
public:
    OrderedGeneSums(int n_stats, int64_t n_genes, int n_slots) :
        n_genes_(n_genes), red_(n_stats * n_genes, n_slots) {}
    double* slot(int s, int stat) { return red_.slot(s) + (size_t)stat * n_genes_; }
    OrderedReducer& reducer() { return red_; }
    std::vector<double> take(int stat) const {
        const double* p = red_.total().data() + (size_t)stat * n_genes_;
        return std::vector<double>(p, p + n_genes_);
    }

private:
    int64_t n_genes_;
    OrderedReducer red_;
};

inline int default_slots() { return 2 * omp_get_max_threads() + 2; }

// Run work(b, slot) for b in [0, nb) on all threads.  With a reducer, block b
// accumulates into slot b % W and slots are merged strictly in block order;
// a block only starts once its slot has been merged.
template <class Work>
void ordered_parallel(int64_t nb, OrderedReducer* red, Work&& work) {
    const int W = red ? red->n_slots() : 1;
    std::atomic<int64_t> next{0};
    std::mutex mu;
    std::condition_variable cv;
    int64_t merged = 0;
    std::vector<uint8_t> done(red ? nb : 0, 0);
    std::exception_ptr err = nullptr;
    std::atomic<bool> failed{false};

#pragma omp parallel
    {
        for (;;) {
            const int64_t b = next.fetch_add(1);
            if (b >= nb || failed.load()) break;
            if (red) {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return b < merged + W || failed.load(); });
                if (failed.load()) break;
            }
            try {
                work(b, red ? (int)(b % W) : 0);
            } catch (...) {
                std::lock_guard<std::mutex> lk(mu);
                if (!err) err = std::current_exception();
                failed.store(true);
                cv.notify_all();
                break;
            }
            if (red) {
                std::lock_guard<std::mutex> lk(mu);
                done[b] = 1;
                while (merged < nb && done[merged]) {
                    red->merge((int)(merged % W));
                    ++merged;
                }
                cv.notify_all();
            }
        }
    }
    if (err) std::rethrow_exception(err);
}

// Decode every block of `src` (in cache-sized row runs) and call
// fn(block_index, slot, rows, ctx) for each run.
template <class V, class Fn>
void for_each_block(const CSRSource& src, OrderedReducer* red, Fn&& fn) {
    const int nt = omp_get_max_threads();
    std::vector<std::unique_ptr<ThreadCtx<V>>> ctx(nt);
    for (auto& c : ctx) c = std::make_unique<ThreadCtx<V>>();
    ordered_parallel(src.n_blocks(), red, [&](int64_t b, int slot) {
        ThreadCtx<V>& t = *ctx[omp_get_thread_num()];
        const int64_t r_end = src.block_start[b + 1];
        int64_t r0 = src.block_start[b];
        while (r0 < r_end) {
            int64_t r1 = r0 + 1;
            while (r1 < r_end && src.indptr[r1 + 1] - src.indptr[r0] <= kSubBlockElems) ++r1;
            load_rows(src, r0, r1, t);
            fn(b, slot, t.blk);
            r0 = r1;
        }
    });
}

// Like for_each_block, but each task decodes `group` consecutive blocks into
// one reduction slot, so a large slot (e.g. groups x genes) is merged once per
// `group` blocks instead of once per block.  `group` must not depend on the
// thread count; the result then does not either.
template <class V, class Fn>
void for_each_block_grouped(const CSRSource& src, OrderedReducer* red, int64_t group, Fn&& fn) {
    if (group < 1) group = 1;
    const int nt = omp_get_max_threads();
    std::vector<std::unique_ptr<ThreadCtx<V>>> ctx(nt);
    for (auto& c : ctx) c = std::make_unique<ThreadCtx<V>>();
    const int64_t nb = src.n_blocks();
    const int64_t ns = (nb + group - 1) / group;
    ordered_parallel(ns, red, [&](int64_t sb, int slot) {
        ThreadCtx<V>& t = *ctx[omp_get_thread_num()];
        const int64_t b_end = std::min(nb, (sb + 1) * group);
        for (int64_t b = sb * group; b < b_end; ++b) {
            const int64_t r_end = src.block_start[b + 1];
            int64_t r0 = src.block_start[b];
            while (r0 < r_end) {
                int64_t r1 = r0 + 1;
                while (r1 < r_end && src.indptr[r1 + 1] - src.indptr[r0] <= kSubBlockElems) ++r1;
                load_rows(src, r0, r1, t);
                fn(b, slot, t.blk);
                r0 = r1;
            }
        }
    });
}

}  // namespace aof
