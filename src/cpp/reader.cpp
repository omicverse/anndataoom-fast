#include "reader.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <stdexcept>

#include "libdeflate.h"

namespace aof {

std::size_t dtype_size(DType t) {
    switch (t) {
        case DType::F32: case DType::I32: case DType::U32: return 4;
        case DType::F64: case DType::I64: case DType::U64: return 8;
        case DType::I16: case DType::U16: return 2;
        case DType::I8: case DType::U8: return 1;
    }
    throw std::invalid_argument("unknown dtype");
}

ReadScratch::ReadScratch() : inflater(libdeflate_alloc_decompressor()) {
    if (!inflater) throw std::bad_alloc();
}

ReadScratch::~ReadScratch() {
    if (inflater) libdeflate_free_decompressor(inflater);
}

ReadScratch::ReadScratch(ReadScratch&& o) noexcept
    : raw(std::move(o.raw)), tmp(std::move(o.tmp)), chunk(std::move(o.chunk)),
      native(std::move(o.native)), cached_chunk(o.cached_chunk),
      cached_owner(o.cached_owner), inflater(o.inflater) {
    o.inflater = nullptr;
}

// Files are opened read-only and read with positional reads, which are safe
// to issue from many threads on one descriptor (pread / ReadFile+OVERLAPPED).
#ifdef _WIN32
static intptr_t open_ro(const std::string& path) {
    int n = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring w(n > 0 ? n : 1, L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &w[0], n);
    HANDLE h = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                           nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (h == INVALID_HANDLE_VALUE)
        throw std::runtime_error("cannot open " + path + ": Windows error " +
                                 std::to_string(GetLastError()));
    return reinterpret_cast<intptr_t>(h);
}

static void close_fd(intptr_t fd) { CloseHandle(reinterpret_cast<HANDLE>(fd)); }
#else
static intptr_t open_ro(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
#ifdef POSIX_FADV_SEQUENTIAL
    ::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
    return fd;
}

static void close_fd(intptr_t fd) { ::close((int)fd); }
#endif

Dataset::Dataset(std::string path, DType dtype, int64_t n_elems, int64_t offset)
    : path_(std::move(path)), dtype_(dtype), esz_(dtype_size(dtype)),
      n_elems_(n_elems), chunked_(false), offset_(offset) {
    fd_ = open_ro(path_);
}

Dataset::Dataset(std::string path, DType dtype, int64_t n_elems, int64_t chunk_elems,
                 std::vector<int64_t> chunk_addr, std::vector<int64_t> chunk_nbytes,
                 std::vector<uint32_t> chunk_mask, std::vector<int> filters)
    : path_(std::move(path)), dtype_(dtype), esz_(dtype_size(dtype)),
      n_elems_(n_elems), chunked_(true), chunk_elems_(chunk_elems),
      chunk_addr_(std::move(chunk_addr)), chunk_nbytes_(std::move(chunk_nbytes)),
      chunk_mask_(std::move(chunk_mask)), filters_(std::move(filters)) {
    if (chunk_elems_ <= 0) throw std::invalid_argument("chunk_elems must be > 0");
    const int64_t n_chunks = (n_elems_ + chunk_elems_ - 1) / chunk_elems_;
    if ((int64_t)chunk_addr_.size() != n_chunks || (int64_t)chunk_nbytes_.size() != n_chunks ||
        (int64_t)chunk_mask_.size() != n_chunks)
        throw std::invalid_argument("chunk table size does not match dataset size");
    for (int f : filters_)
        if (f != H5Z_DEFLATE && f != H5Z_SHUFFLE && f != H5Z_FLETCHER32)
            throw std::invalid_argument("unsupported HDF5 filter id " + std::to_string(f));
    fd_ = open_ro(path_);
}

Dataset::Dataset(const void* data, DType dtype, int64_t n_elems)
    : dtype_(dtype), esz_(dtype_size(dtype)), n_elems_(n_elems), chunked_(false),
      mem_(static_cast<const unsigned char*>(data)) {
    if (!data && n_elems) throw std::invalid_argument("null buffer");
}

Dataset::~Dataset() {
    if (fd_ != -1) close_fd(fd_);
}

void Dataset::pread_full(unsigned char* buf, std::size_t n, int64_t off) const {
#ifdef _WIN32
    HANDLE h = reinterpret_cast<HANDLE>(fd_);
    while (n > 0) {
        DWORD want = (DWORD)std::min<std::size_t>(n, 1u << 30), got = 0;
        OVERLAPPED ov{};
        ov.Offset = (DWORD)((uint64_t)off & 0xffffffffu);
        ov.OffsetHigh = (DWORD)((uint64_t)off >> 32);
        if (!ReadFile(h, buf, want, &got, &ov))
            throw std::runtime_error("ReadFile failed on " + path_ + ": Windows error " +
                                     std::to_string(GetLastError()));
        if (got == 0) throw std::runtime_error("unexpected end of file in " + path_);
        buf += got;
        n -= got;
        off += got;
    }
#else
    while (n > 0) {
        ssize_t r = ::pread((int)fd_, buf, n, (off_t)off);
        if (r < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("pread failed on " + path_ + ": " + std::strerror(errno));
        }
        if (r == 0) throw std::runtime_error("unexpected end of file in " + path_);
        buf += r;
        n -= (std::size_t)r;
        off += r;
    }
#endif
}

static void unshuffle(const unsigned char* in, unsigned char* out, std::size_t nbytes,
                      std::size_t esz) {
    // HDF5 shuffle stores byte k of every element contiguously; trailing bytes
    // that do not form a whole element are left as-is.
    const std::size_t n = nbytes / esz;
    for (std::size_t k = 0; k < esz; ++k) {
        const unsigned char* src = in + k * n;
        for (std::size_t i = 0; i < n; ++i) out[i * esz + k] = src[i];
    }
    std::memcpy(out + n * esz, in + n * esz, nbytes - n * esz);
}

void Dataset::decode_chunk(int64_t c, ReadScratch& s) const {
    if (s.cached_owner == this && s.cached_chunk == c) return;
    const std::size_t full = (std::size_t)chunk_elems_ * esz_;
    s.chunk.resize(full);
    const int64_t addr = chunk_addr_[c];
    if (addr < 0) {  // unallocated chunk -> fill value (0)
        std::memset(s.chunk.data(), 0, full);
    } else {
        const std::size_t nb = (std::size_t)chunk_nbytes_[c];
        s.raw.resize(nb);
        pread_full(s.raw.data(), nb, addr);
        // Walk the pipeline backwards; `cur` points at the current bytes.
        std::vector<unsigned char>* cur = &s.raw;
        std::size_t cur_n = nb;
        const uint32_t mask = chunk_mask_[c];
        for (int i = (int)filters_.size() - 1; i >= 0; --i) {
            if (mask & (1u << i)) continue;
            const int f = filters_[i];
            if (f == H5Z_FLETCHER32) {
                if (cur_n < 4) throw std::runtime_error("corrupt fletcher32 chunk");
                cur_n -= 4;
            } else if (f == H5Z_DEFLATE) {
                std::vector<unsigned char>* dst = (cur == &s.tmp) ? &s.chunk : &s.tmp;
                dst->resize(std::max(full, dst->size()));
                std::size_t got = 0;
                libdeflate_result r = libdeflate_zlib_decompress(
                    s.inflater, cur->data(), cur_n, dst->data(), dst->size(), &got);
                if (r != LIBDEFLATE_SUCCESS)
                    throw std::runtime_error("deflate decode failed in " + path_);
                cur = dst;
                cur_n = got;
            } else if (f == H5Z_SHUFFLE) {
                std::vector<unsigned char>* dst = (cur == &s.tmp) ? &s.chunk : &s.tmp;
                dst->resize(std::max(cur_n, dst->size()));
                unshuffle(cur->data(), dst->data(), cur_n, esz_);
                cur = dst;
            }
        }
        if (cur_n < full) throw std::runtime_error("decoded chunk too short in " + path_);
        if (cur != &s.chunk) {
            s.chunk.resize(full);
            std::memcpy(s.chunk.data(), cur->data(), full);
        }
    }
    s.cached_owner = this;
    s.cached_chunk = c;
}

void Dataset::read_native(int64_t e0, int64_t e1, unsigned char* out, ReadScratch& s) const {
    if (e0 < 0 || e1 > n_elems_ || e0 > e1) throw std::out_of_range("dataset range");
    if (e0 == e1) return;
    if (mem_) {
        std::memcpy(out, mem_ + e0 * (int64_t)esz_, (std::size_t)(e1 - e0) * esz_);
        return;
    }
    if (!chunked_) {
        pread_full(out, (std::size_t)(e1 - e0) * esz_, offset_ + e0 * (int64_t)esz_);
        return;
    }
    const int64_t c0 = e0 / chunk_elems_, c1 = (e1 - 1) / chunk_elems_;
    for (int64_t c = c0; c <= c1; ++c) {
        decode_chunk(c, s);
        const int64_t cs = c * chunk_elems_;
        const int64_t a = std::max(e0, cs), b = std::min(e1, cs + chunk_elems_);
        std::memcpy(out + (a - e0) * esz_, s.chunk.data() + (a - cs) * esz_,
                    (std::size_t)(b - a) * esz_);
    }
}

template <class Src, class T>
static void convert(const unsigned char* in, T* out, int64_t n) {
    const Src* p = reinterpret_cast<const Src*>(in);
    for (int64_t i = 0; i < n; ++i) out[i] = static_cast<T>(p[i]);
}

template <class T>
static void convert_any(DType dt, const unsigned char* in, T* out, int64_t n) {
    switch (dt) {
        case DType::F32: convert<float>(in, out, n); break;
        case DType::F64: convert<double>(in, out, n); break;
        case DType::I8: convert<int8_t>(in, out, n); break;
        case DType::I16: convert<int16_t>(in, out, n); break;
        case DType::I32: convert<int32_t>(in, out, n); break;
        case DType::I64: convert<int64_t>(in, out, n); break;
        case DType::U8: convert<uint8_t>(in, out, n); break;
        case DType::U16: convert<uint16_t>(in, out, n); break;
        case DType::U32: convert<uint32_t>(in, out, n); break;
        case DType::U64: convert<uint64_t>(in, out, n); break;
    }
}

// Decoded chunks are converted straight into `out` while still in cache;
// there is no intermediate native-dtype copy of the whole range.
template <class T>
void Dataset::read_as(int64_t e0, int64_t e1, T* out, ReadScratch& s) const {
    if (e0 < 0 || e1 > n_elems_ || e0 > e1) throw std::out_of_range("dataset range");
    if (e0 == e1) return;
    if (mem_) {
        convert_any(dtype_, mem_ + e0 * (int64_t)esz_, out, e1 - e0);
        return;
    }
    if (!chunked_) {
        s.native.resize((std::size_t)(e1 - e0) * esz_);
        read_native(e0, e1, s.native.data(), s);
        convert_any(dtype_, s.native.data(), out, e1 - e0);
        return;
    }
    const int64_t c0 = e0 / chunk_elems_, c1 = (e1 - 1) / chunk_elems_;
    for (int64_t c = c0; c <= c1; ++c) {
        decode_chunk(c, s);
        const int64_t cs = c * chunk_elems_;
        const int64_t a = std::max(e0, cs), b = std::min(e1, cs + chunk_elems_);
        convert_any(dtype_, s.chunk.data() + (a - cs) * esz_, out + (a - e0), b - a);
    }
}

template <class T>
void Dataset::read_parallel(int64_t e0, int64_t e1, T* out, int n_threads) const {
    if (e0 < 0 || e1 > n_elems_ || e0 > e1) throw std::out_of_range("dataset range");
    if (e0 == e1) return;
    // Pieces: one HDF5 chunk each, or 4 Mi elements of contiguous data.
    const int64_t piece = chunked_ ? chunk_elems_ : (int64_t)1 << 22;
    const int64_t p0 = e0 / piece, p1 = (e1 - 1) / piece;
    if (n_threads <= 1 || p0 == p1 || mem_) {
        ReadScratch s;
        read_as(e0, e1, out, s);
        return;
    }
    std::string err;
#pragma omp parallel num_threads(n_threads)
    {
        ReadScratch s;
#pragma omp for schedule(dynamic, 1)
        for (int64_t p = p0; p <= p1; ++p) {
            const int64_t a = std::max(e0, p * piece), b = std::min(e1, (p + 1) * piece);
            try {
                read_as(a, b, out + (a - e0), s);
            } catch (const std::exception& ex) {
#pragma omp critical(aof_read_err)
                if (err.empty()) err = ex.what();
            }
        }
    }
    if (!err.empty()) throw std::runtime_error(err);
}

template void Dataset::read_as<double>(int64_t, int64_t, double*, ReadScratch&) const;
template void Dataset::read_as<float>(int64_t, int64_t, float*, ReadScratch&) const;
template void Dataset::read_as<int32_t>(int64_t, int64_t, int32_t*, ReadScratch&) const;
template void Dataset::read_as<int64_t>(int64_t, int64_t, int64_t*, ReadScratch&) const;

template void Dataset::read_parallel<double>(int64_t, int64_t, double*, int) const;
template void Dataset::read_parallel<float>(int64_t, int64_t, float*, int) const;
template void Dataset::read_parallel<int32_t>(int64_t, int64_t, int32_t*, int) const;
template void Dataset::read_parallel<int64_t>(int64_t, int64_t, int64_t*, int) const;

}  // namespace aof
