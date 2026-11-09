// Parallel reader for HDF5 1-D datasets that bypasses libhdf5.
//
// libhdf5 serialises every read behind a global lock, so a multithreaded
// pipeline cannot read through it.  Instead, the Python side uses h5py once to
// describe the dataset's on-disk layout (contiguous byte offset, or the byte
// address/size/filter-mask of every chunk), and this reader fetches raw bytes
// with pread() and decodes the HDF5 filter pipeline itself (deflate, shuffle,
// fletcher32).  pread() is thread-safe, so any number of threads can decode
// different ranges of the same dataset concurrently.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct libdeflate_decompressor;

namespace aof {

enum class DType : int {
    F32 = 0, F64 = 1,
    I8 = 2, I16 = 3, I32 = 4, I64 = 5,
    U8 = 6, U16 = 7, U32 = 8, U64 = 9,
};

std::size_t dtype_size(DType t);

// HDF5 filter identifiers (H5Zpublic.h).
constexpr int H5Z_DEFLATE = 1;
constexpr int H5Z_SHUFFLE = 2;
constexpr int H5Z_FLETCHER32 = 3;

// Per-thread decode buffers.  One per worker thread; never shared.
struct ReadScratch {
    std::vector<unsigned char> raw;     // compressed chunk bytes
    std::vector<unsigned char> tmp;     // intermediate filter output
    std::vector<unsigned char> chunk;   // fully decoded chunk
    std::vector<unsigned char> native;  // native-dtype staging for conversion
    int64_t cached_chunk = -1;          // id of the chunk held in `chunk`
    const void* cached_owner = nullptr; // dataset that `chunk` belongs to
    libdeflate_decompressor* inflater = nullptr;
    ReadScratch();
    ~ReadScratch();
    ReadScratch(const ReadScratch&) = delete;
    ReadScratch& operator=(const ReadScratch&) = delete;
    ReadScratch(ReadScratch&&) noexcept;
};

class Dataset {
public:
    // Contiguous, unfiltered dataset stored at byte `offset`.
    Dataset(std::string path, DType dtype, int64_t n_elems, int64_t offset);
    // Chunked dataset.  `chunk_addr[c]` is the file address of chunk c
    // (-1 if never written -> fill value 0), `chunk_nbytes[c]` its stored size
    // and `chunk_mask[c]` the per-chunk filter mask (bit i set = filter i
    // skipped).  `filters` lists the pipeline in write order.
    Dataset(std::string path, DType dtype, int64_t n_elems, int64_t chunk_elems,
            std::vector<int64_t> chunk_addr, std::vector<int64_t> chunk_nbytes,
            std::vector<uint32_t> chunk_mask, std::vector<int> filters);
    // In-memory array (e.g. a numpy buffer); the caller keeps it alive.
    Dataset(const void* data, DType dtype, int64_t n_elems);
    ~Dataset();
    Dataset(const Dataset&) = delete;
    Dataset& operator=(const Dataset&) = delete;

    DType dtype() const { return dtype_; }
    int64_t size() const { return n_elems_; }
    bool chunked() const { return chunked_; }
    bool in_memory() const { return mem_ != nullptr; }

    // Read elements [e0, e1) converted to T.  Thread-safe given a private
    // scratch.
    template <class T>
    void read_as(int64_t e0, int64_t e1, T* out, ReadScratch& s) const;

private:
    void read_native(int64_t e0, int64_t e1, unsigned char* out, ReadScratch& s) const;
    void decode_chunk(int64_t c, ReadScratch& s) const;
    void pread_full(unsigned char* buf, std::size_t n, int64_t off) const;

    std::string path_;
    intptr_t fd_ = -1;  // POSIX fd or Windows HANDLE
    DType dtype_;
    std::size_t esz_;
    int64_t n_elems_;
    bool chunked_;
    int64_t offset_ = -1;
    int64_t chunk_elems_ = 0;
    std::vector<int64_t> chunk_addr_, chunk_nbytes_;
    std::vector<uint32_t> chunk_mask_;
    std::vector<int> filters_;
    const unsigned char* mem_ = nullptr;
};

}  // namespace aof
