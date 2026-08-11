/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "page_decode.cuh"

#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda/pipeline>

#include <climits>

namespace cudf::io::parquet::detail {

// DELTA_XXX encoding support
//
// DELTA_BINARY_PACKED is used for INT32 and INT64 data types. Encoding begins with a header
// containing a block size, number of mini-blocks in each block, total value count, and first
// value. The first three are ULEB128 variable length ints, and the last is a zigzag ULEB128
// varint.
//   -- the block size is a multiple of 128
//   -- the mini-block count is chosen so that each mini-block will contain a multiple of 32 values
//   -- the value count includes the first value stored in the header
//
// It seems most Parquet encoders will stick with a block size of 128, and 4 mini-blocks of 32
// elements each. arrow-rs will use a block size of 256 for 64-bit ints.
//
// Following the header are the data blocks. Each block is further divided into mini-blocks, with
// each mini-block having its own encoding bitwidth. Each block begins with a header containing a
// zigzag ULEB128 encoded minimum delta value, followed by an array of uint8 bitwidths, one entry
// per mini-block. While encoding, the lowest delta value is subtracted from all the deltas in the
// block to ensure that all encoded values are positive. The deltas for each mini-block are bit
// packed using the same encoding as the RLE/Bit-Packing Hybrid encoder.

// The DELTA_BINARY_PACKED spec requires the number of values in a mini-block to be a multiple of
// 32. The decoders rely on the coincidence that this also equals warp size; they produce values
// in warp_size-wide passes, so it must divide every spec-valid mini-block size.
constexpr int delta_mini_block_size_multiple = 32;
static_assert(delta_mini_block_size_multiple % cudf::detail::warp_size == 0,
              "warp_size must divide the DELTA mini-block size multiple; the pass-based decoders "
              "assume warp_size divides every spec-valid mini-block size");

// The decode loops produce up to two (warp_size-wide) passes per iteration: pages whose
// mini-blocks hold at least two passes keep the two-pass batch the loops have always used, and
// running several passes back to back amortizes the per-iteration synchronization.
constexpr int delta_max_batch_size = 2 * cudf::detail::warp_size;

// The rolling buffer must hold two batches in flight (the consumer drains one batch while the
// producer decodes the next), plus one slot for the first value from the block header: it is not
// stored in the buffer, but it still impacts buffer indexing and we need to account for it to
// avoid race conditions.
constexpr int delta_rolling_buf_size = (2 * delta_max_batch_size) + 1;

/**
 * @brief Read a ULEB128 varint integer
 *
 * @param[in,out] cur The current data position, updated after the read
 * @param[in] end The end data position
 *
 * @return The value read
 */
inline __device__ uleb128_t get_uleb128(uint8_t const*& cur, uint8_t const* end)
{
  uleb128_t v = 0, l = 0, c;
  while (cur < end) {
    c = *cur++;
    v |= (c & 0x7f) << l;
    l += 7;
    if ((c & 0x80) == 0) { return v; }
  }
  return v;
}

/**
 * @brief Read a ULEB128 zig-zag encoded varint integer
 *
 * @param[in,out] cur The current data position, updated after the read
 * @param[in] end The end data position
 *
 * @return The value read
 */
inline __device__ zigzag128_t get_zz128(uint8_t const*& cur, uint8_t const* end)
{
  uleb128_t u = get_uleb128(cur, end);
  return static_cast<zigzag128_t>((u >> 1u) ^ -static_cast<zigzag128_t>(u & 1));
}

/**
 * @brief Extract one bit-packed field from a mini-block body.
 *
 * Reads the `width`-bit little-endian field whose least significant bit sits at bit offset
 * `bit_pos` relative to `d_start`. `bit_pos` may be negative: callers position `d_start` past the
 * end of the values they are unpacking and index backwards, so the shift/mask below must floor
 * toward negative infinity (a plain `/` and `%` would round toward zero).
 *
 * Reads are clamped to `end`. A field that would read past it contributes zero bits, matching the
 * byte-at-a-time behaviour this replaces.
 *
 * The common case is served by one (or two) unaligned 64-bit loads instead of up to nine dependent
 * byte loads. Only the final bytes of a block, where a wide load would run past `end`, fall back to
 * the byte loop.
 */
inline __device__ zigzag128_t unpack_bitpacked_field(uint8_t const* d_start,
                                                     uint8_t const* end,
                                                     int32_t bit_pos,
                                                     uint32_t width)
{
  if (width == 0) { return 0; }

  int32_t const ofs = bit_pos & 7;
  uint8_t const* p  = d_start + (bit_pos >> 3);
  if (p >= end) { return 0; }

  // Computed in unsigned to keep the wide-width edges well defined; the caller reinterprets the
  // raw field as a signed delta exactly as the byte loop did.
  uint64_t const mask = (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);

  if (width <= 64) {
    if (ofs + width <= 64) {
      if (p + sizeof(uint64_t) <= end) {
        uint64_t w;
        memcpy(&w, p, sizeof(w));
        return static_cast<zigzag128_t>((w >> ofs) & mask);
      }
    } else if (p + 2 * sizeof(uint64_t) <= end) {
      // ofs > 0 here: ofs + width > 64 with width <= 64 implies ofs >= 1, so the shift below is
      // in [1, 63] and never the undefined shift-by-64.
      uint64_t lo, hi;
      memcpy(&lo, p, sizeof(lo));
      memcpy(&hi, p + sizeof(lo), sizeof(hi));
      return static_cast<zigzag128_t>(((lo >> ofs) | (hi << (64 - ofs))) & mask);
    }
  }

  // Tail of a block (and malformed widths): byte at a time, bounded by `end`.
  uint64_t acc = static_cast<uint64_t>(*p++) >> ofs;
  uint32_t c   = CHAR_BIT - ofs;
  while (c < width && p < end) {
    acc |= static_cast<uint64_t>(*p++) << c;
    c += CHAR_BIT;
  }
  return static_cast<zigzag128_t>(acc & mask);
}

// Bytes copied per cp.async group, and the number of those the ring holds. One group is kept in
// flight, so the ring only needs the chunk being read, the one being copied, and slack; four keeps
// the modular addressing a mask and leaves room for a pass that straddles a chunk boundary.
// Upper bound on values decoded per lane when a warp takes a whole block at a time
// (block_size / warp_size). 128-value blocks give 4; arrow-rs uses 256 for 64-bit ints, giving 8.
// Larger blocks fall back to the pass-at-a-time path.
constexpr int delta_max_values_per_lane = 8;

// Per-SM shared memory differs by a factor of 3.5 across the architectures this TU is built for,
// and the ring is charged against it once per resident block. Sizing it for Hopper would make
// shared the binding occupancy limiter everywhere else:
//
//   arch                        shared/SM   blocks at 3.3 KB   at 7.3 KB
//   T4 (sm_75)                     64 KB           19               8
//   V100 (sm_70)                   96 KB           29              13
//   A10/L40S/RTX 50xx (86/89/120) 100 KB           30              13
//   A100 (sm_80)                  164 KB           32 (capped)     22
//   H100/B200 (sm_90/100)         228 KB           32 (capped)     31
//
// So: full ring on the 228 KB parts, half on A100, and none below that -- sm_70 and sm_75 have no
// cp.async either, and their fallback (a synchronous copy) measured slower than reading global
// directly. Chunk *count* is fixed so the addressing and span guarantees below are unchanged;
// only the chunk size varies.
#if !defined(__CUDA_ARCH__)
#define CUDF_PARQUET_DELTA_RING_CHUNK_BYTES 512  // host pass; device value is what matters
#elif __CUDA_ARCH__ == 900 || (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1100)
#define CUDF_PARQUET_DELTA_RING_CHUNK_BYTES 512  // 228 KB/SM -> 4 KB ring
#elif __CUDA_ARCH__ == 800
#define CUDF_PARQUET_DELTA_RING_CHUNK_BYTES 256  // 164 KB/SM -> 2 KB ring
#else
#define CUDF_PARQUET_DELTA_RING_CHUNK_BYTES 0  // disabled; decode reads global
#endif

constexpr bool delta_ring_enabled = CUDF_PARQUET_DELTA_RING_CHUNK_BYTES > 0;
// Keep the chunk size non-zero even when disabled: the ring's arithmetic is still compiled for
// those architectures (it is just never reached), and a zero divisor is a hard error.
constexpr int delta_ring_chunk_bytes =
  delta_ring_enabled ? CUDF_PARQUET_DELTA_RING_CHUNK_BYTES : 512;
// More, smaller chunks rather than fewer large ones: the span below scales with (chunks - 2), so
// this covers a wider block body for the same total shared.
constexpr int delta_ring_chunks = 8;

// A block body may straddle chunk boundaries, so the ring must hold the chunk the read starts in,
// the one it ends in, and one more in flight. Blocks whose body exceeds this fall back to reading
// global -- on the 512 B configuration that is anything past a ~1 KB body, which the common
// 128-value/4-mini-block shape (~416 B at typical widths) stays well inside.
constexpr int delta_ring_span_bytes =
  delta_ring_enabled ? (delta_ring_chunks - 2) * delta_ring_chunk_bytes : 0;
constexpr int delta_ring_bytes = delta_ring_chunk_bytes * delta_ring_chunks;
// one word when disabled: the struct still has to be a valid type, it is just never used
constexpr int delta_ring_words = delta_ring_enabled ? delta_ring_bytes / sizeof(uint32_t) : 1;

/**
 * @brief A shared-memory ring over one page's encoded bytes, refilled asynchronously.
 *
 * The flat DELTA_BINARY decoder is latency-bound on the mini-block bodies: it reads them as many
 * small, overlapping, L1-mediated loads and never gets enough memory-level parallelism to approach
 * DRAM bandwidth. Copying the same bytes synchronously into shared does not help — it removes
 * transactions but leaves every one of them latency-exposed, and measured 9% *slower*. Issuing the
 * copy for chunk c+1 before chunk c is consumed is what converts the latency into bandwidth.
 *
 * Owned by the caller rather than embedded in `delta_binary_decoder`, so the DELTA_BYTE_ARRAY
 * decoders (which hold two `delta_binary_decoder`s each) do not pay for a ring they do not use.
 *
 * Copies go through `cuda::memcpy_async` on a thread-scope `cuda::pipeline`, which selects the best
 * available copy path (`cp.async` on sm_80+, plain loads below it) rather than hard-coding one, so
 * no architecture guards are needed here. The pipeline is per-thread state and therefore lives in
 * the caller's registers, not in this shared object.
 */
using delta_pipeline = cuda::pipeline<cuda::thread_scope_thread>;

struct delta_stream_ring {
  uint8_t const* page_base;  // 16B-aligned; the wide copy path requires it on both ends
  int page_len;              // valid bytes from page_base
  int issued_chunks;
  int released_chunks;
  bool active;

  // the wide copy path needs 16B alignment; the enclosing struct's layout must not be allowed to
  // shift this to an 8-byte boundary.
  __align__(16) uint32_t buf[delta_ring_words];

  __device__ inline void issue_chunk(
    int c,
    cg::thread_block_tile<cudf::detail::warp_size, cg::thread_block> const& warp,
    delta_pipeline& pipe)
  {
    using cudf::detail::warp_size;
    if constexpr (not delta_ring_enabled) { return; }  // `buf` is a stub on these architectures
    int const slot        = c & (delta_ring_chunks - 1);
    auto* const dst_base  = reinterpret_cast<uint8_t*>(buf) + slot * delta_ring_chunk_bytes;
    auto const* const src = page_base + static_cast<size_t>(c) * delta_ring_chunk_bytes;
    int const chunk_start = c * delta_ring_chunk_bytes;

    pipe.producer_acquire();
    for (int i = warp.thread_rank(); i < delta_ring_chunk_bytes / 16; i += warp_size) {
      int const off    = i * 16;
      int const remain = page_len - (chunk_start + off);
      auto* const dst  = dst_base + off;
      if (remain >= 16) {
        cuda::memcpy_async(dst, src + off, cuda::aligned_size_t<16>(16), pipe);
      } else {
        // Past the end of the page. memcpy_async has no short-source zero-fill, so do it here;
        // this is one 16B unit per page and keeps the ring's out-of-range bytes reading as zero,
        // matching the clamping the direct-from-global path does.
        for (int k = 0; k < 16; ++k) {
          dst[k] = (k < remain) ? src[off + k] : uint8_t{0};
        }
      }
    }
    pipe.producer_commit();
  }

  /**
   * @brief Point the ring at a page and prime the pipeline. Called by the whole warp.
   */
  __device__ inline void attach(
    uint8_t const* data_start,
    uint8_t const* data_end,
    cg::thread_block_tile<cudf::detail::warp_size, cg::thread_block> const& warp,
    delta_pipeline& pipe)
  {
    if constexpr (not delta_ring_enabled) {
      active = false;
      return;
    }
    auto const misalign = static_cast<int>(reinterpret_cast<uintptr_t>(data_start) & 15);
    page_base           = data_start - misalign;
    page_len            = static_cast<int>(data_end - page_base);
    issued_chunks       = 0;
    released_chunks     = 0;
    active              = true;
    issue_chunk(0, warp, pipe);
    issue_chunk(1, warp, pipe);
    issued_chunks = 2;
  }

  /// Ensure bytes up to `end_off` (relative to `page_base`) have landed, keeping one chunk in
  /// flight behind them. Called by the whole warp on a warp-uniform condition.
  __device__ inline void reach(
    int end_off,
    cg::thread_block_tile<cudf::detail::warp_size, cg::thread_block> const& warp,
    delta_pipeline& pipe)
  {
    int const need = end_off / delta_ring_chunk_bytes;
    while (issued_chunks <= need + 1) {
      issue_chunk(issued_chunks, warp, pipe);
      issued_chunks++;
    }
    // Batches are committed in chunk order, so draining up to `need` leaves exactly the chunks
    // after it in flight.
    while (released_chunks <= need) {
      pipe.consumer_wait();
      pipe.consumer_release();
      released_chunks++;
    }
    warp.sync();
  }

  __device__ inline uint32_t word(int i) const { return buf[i & (delta_ring_words - 1)]; }

  /// Extract the `width`-bit field at bit offset `bit` relative to `page_base`.
  __device__ inline zigzag128_t extract(int bit, uint32_t width) const
  {
    if (width == 0) { return 0; }
    int const w   = bit >> 5;
    int const ofs = bit & 31;
    uint64_t v =
      (static_cast<uint64_t>(word(w)) | (static_cast<uint64_t>(word(w + 1)) << 32)) >> ofs;
    // ofs == 0 can only need two words (width <= 64), so this shift is always in [1, 63]
    if (ofs != 0 && width + ofs > 64) { v |= static_cast<uint64_t>(word(w + 2)) << (64 - ofs); }
    uint64_t const mask = (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
    return static_cast<zigzag128_t>(v & mask);
  }
};

struct delta_binary_decoder {
  uint8_t const* block_start;  // start of data, but updated as data is read
  uint8_t const* block_end;    // end of data
  uleb128_t block_size;        // usually 128, must be multiple of 128
  uleb128_t mini_block_count;  // usually 4, chosen such that block_size/mini_block_count is a
                               // multiple of 32
  uleb128_t value_count;       // total values encoded in the block
  zigzag128_t first_value;     // initial value, stored in the header
  zigzag128_t last_value;      // last value decoded

  uint32_t values_per_mb;      // block_size / mini_block_count, must be multiple of 32
  uint32_t current_value_idx;  // current value index, initialized to 0 at start of block
  uint32_t cur_pass;           // current pass within the mini-block

  zigzag128_t cur_min_delta;     // min delta for the block
  uint32_t cur_mb;               // index of the current mini-block within the block
  uint8_t const* cur_mb_start;   // pointer to the start of the current mini-block data
  uint8_t const* cur_bitwidths;  // pointer to the bitwidth array in the block
  bool error;                    // flag to catch malformed headers

  zigzag128_t value[delta_rolling_buf_size];  // circular buffer of delta values

  // returns the value stored in the `value` array at index
  // `rolling_index<delta_rolling_buf_size>(idx)`. If `idx` is `0`, then return `first_value`.
  __device__ constexpr zigzag128_t value_at(size_type idx)
  {
    return idx == 0 ? first_value : value[rolling_index<delta_rolling_buf_size>(idx)];
  }

  // returns the number of values encoded in the block data. when all_values is true,
  // account for the first value in the header. otherwise just count the values encoded
  // in the mini-block data.
  __device__ constexpr uint32_t num_encoded_values(bool all_values)
  {
    return value_count == 0 ? 0 : all_values ? value_count : value_count - 1;
  }

  // index just past the values decode_next_pass() has produced so far (0 before the first pass,
  // even though the header value already occupies index 0)
  __device__ uint32_t next_pass_start_idx()
  {
    return current_value_idx + cur_pass * cudf::detail::warp_size;
  }

  // read mini-block header into state object. should only be called from init_binary_block or
  // setup_next_mini_block. header format is:
  //
  // | min delta (int) | bit-width array (1 byte * mini_block_count) |
  //
  // on exit db->cur_mb is 0 and db->cur_mb_start points to the first mini-block of data, or
  // nullptr if out of data.
  // is_decode indicates whether this is being called from initialization code (false) or
  // the actual decoding (true)
  inline __device__ void init_mini_block(bool is_decode)
  {
    cur_mb       = 0;
    cur_mb_start = nullptr;

    if (current_value_idx < num_encoded_values(is_decode)) {
      auto d_start  = block_start;
      cur_min_delta = get_zz128(d_start, block_end);
      cur_bitwidths = d_start;

      d_start += mini_block_count;
      cur_mb_start = d_start;
    }
  }

  // read delta binary header into state object. should be called on thread 0. header format is:
  //
  // | block size (uint) | mini-block count (uint) | value count (uint) | first value (int) |
  //
  // also initializes the first mini-block before exit
  inline __device__ void init_binary_block(uint8_t const* d_start, uint8_t const* d_end)
  {
    block_end        = d_end;
    block_size       = get_uleb128(d_start, d_end);
    mini_block_count = get_uleb128(d_start, d_end);
    value_count      = get_uleb128(d_start, d_end);
    first_value      = get_zz128(d_start, d_end);
    last_value       = first_value;

    current_value_idx = 0;
    cur_pass          = 0;
    error             = false;

    // Validate the header against the DELTA_BINARY_PACKED spec: the mini-block count must evenly
    // divide the block size, and each mini-block must hold a multiple of 32 values. The decoders
    // rely on the latter to advance from one mini-block to the next.
    if (mini_block_count == 0 or block_size == 0 or (block_size % mini_block_count) != 0 or
        ((block_size / mini_block_count) % delta_mini_block_size_multiple) != 0) {
      error         = true;
      value_count   = 0;
      values_per_mb = 1;
      block_start   = d_end;
      cur_mb        = 0;
      cur_mb_start  = d_end;
      cur_bitwidths = d_end;
      return;
    }

    values_per_mb = block_size / mini_block_count;

    // init the first mini-block
    block_start = d_start;

    // only call init if there are actually encoded values
    if (value_count > 1) { init_mini_block(false); }
  }

  // skip to the start of the next mini-block. should only be called on thread 0.
  // calls init_binary_block if currently on the last mini-block in a block.
  // is_decode indicates whether this is being called from initialization code (false) or
  // the actual decoding (true)
  inline __device__ void setup_next_mini_block(bool is_decode)
  {
    if (current_value_idx >= num_encoded_values(is_decode)) { return; }

    current_value_idx += values_per_mb;

    // just set pointer to start of next mini_block
    if (cur_mb < mini_block_count - 1) {
      cur_mb_start += cur_bitwidths[cur_mb] * values_per_mb / CHAR_BIT;
      cur_mb++;
    }
    // out of mini-blocks, start a new block
    else {
      block_start = cur_mb_start + cur_bitwidths[cur_mb] * values_per_mb / CHAR_BIT;
      init_mini_block(is_decode);
    }
  }

  // given start/end pointers in the data, find the end of the binary encoded block. when done,
  // `this` will be initialized with the correct start and end positions. returns the end, which is
  // start of data/next block. should only be called from thread 0.
  inline __device__ uint8_t const* find_end_of_block(uint8_t const* start, uint8_t const* end)
  {
    // read block header
    init_binary_block(start, end);

    // test for no encoded values. a single value will be in the block header.
    if (value_count <= 1) { return block_start; }

    // read mini-block headers and skip over data
    while (current_value_idx < num_encoded_values(false)) {
      setup_next_mini_block(false);
    }
    // calculate the correct end of the block
    auto const* const new_end = cur_mb == 0 ? block_start : cur_mb_start;
    // re-init block with correct end
    init_binary_block(start, new_end);
    return new_end;
  }

  // account for the first value from the block header before the first mini-block is decoded.
  // the first value is not encoded in the mini-block data, but it still occupies index 0 of the
  // value stream. returns true if there are more values to decode after the header value.
  // called by all threads in a single warp `warp`.
  inline __device__ bool advance_past_first_value(
    cg::thread_block_tile<cudf::detail::warp_size, cg::thread_block> const& warp)
  {
    if (current_value_idx >= value_count) { return false; }

    if (current_value_idx == 0) {
      // make sure all threads access current_value_idx above before incrementing
      warp.sync();
      if (warp.thread_rank() == 0) { current_value_idx++; }
      warp.sync();
      if (current_value_idx >= value_count) { return false; }
    }
    return true;
  }

  // decode a single warp_size-wide pass (indexed by `pass`) of the current mini-block and convert
  // the deltas to values, returning this lane's value rather than publishing it to the rolling
  // buffer. called by all threads in a single warp `warp`.
  //
  // On return `last_value` has been advanced and published to the whole warp, so the caller may
  // immediately start the next pass. Callers that need the value in the rolling buffer should use
  // calc_mini_block_pass() instead.
  inline __device__ zigzag128_t calc_mini_block_pass_value(
    uint32_t pass,
    cg::thread_block_tile<cudf::detail::warp_size, cg::thread_block> const& warp,
    delta_stream_ring* ring = nullptr,
    delta_pipeline* pipe    = nullptr)
  {
    using cudf::detail::warp_size;

    auto const lane_id     = static_cast<int>(warp.thread_rank());
    uint32_t const mb_bits = cur_bitwidths[cur_mb];

    // byte span of this pass's values, and the position just past them: the unpack indexes
    // backwards from there so that lane offsets stay non-positive
    int const body_bytes  = static_cast<int>(warp_size * mb_bits / CHAR_BIT);
    auto const pass_start = cur_mb_start + pass * body_bytes;
    auto const d_start    = pass_start + body_bytes;

    // mb_bits comes from shared state, so this is warp-uniform: the ring walk may not sit inside
    // the divergent per-lane range test below.
    int const pass_off =
      (ring != nullptr && ring->active) ? static_cast<int>(pass_start - ring->page_base) : -1;
    bool const use_ring = (mb_bits <= 64) && (pass_off >= 0);
    if (use_ring) { ring->reach(pass_off + body_bytes + 8, warp, *pipe); }

    zigzag128_t delta = 0;
    if (current_value_idx + pass * warp_size + lane_id < value_count) {
      delta =
        use_ring
          ? ring->extract(pass_off * CHAR_BIT + lane_id * static_cast<int>(mb_bits), mb_bits)
          : unpack_bitpacked_field(
              d_start, block_end, (lane_id - warp_size) * static_cast<int32_t>(mb_bits), mb_bits);
    }

    // add min delta to get true delta
    delta += cur_min_delta;

    // do inclusive scan to get value - first_value at each position. cg::inclusive_scan is
    // shuffle-based and carries no shared storage, so any number of delta decoders (e.g. the
    // prefix and suffix decoder warps of the DELTA_BYTE_ARRAY kernels) can run it concurrently,
    // each over its own warp tile, with no risk of aliasing.
    delta = cg::inclusive_scan(warp, delta, cg::plus<int64_t>{});

    // now add first value from header or last value from previous pass to get true value
    delta += last_value;

    // save value from last lane in warp. this will become the 'first value' added to the
    // deltas calculated in the next pass (or invocation).
    if (lane_id == warp_size - 1) { last_value = delta; }
    warp.sync();
    return delta;
  }

  // decode a single warp_size-wide pass (indexed by `pass`) of the current mini-block and publish
  // the values to the rolling buffer (see decode_next_pass). called by all threads in a single
  // warp `warp`.
  inline __device__ void calc_mini_block_pass(
    uint32_t pass, cg::thread_block_tile<cudf::detail::warp_size, cg::thread_block> const& warp)
  {
    using cudf::detail::warp_size;

    auto const lane_id = static_cast<int>(warp.thread_rank());
    auto const delta   = calc_mini_block_pass_value(pass, warp);
    int const value_idx =
      rolling_index<delta_rolling_buf_size>(current_value_idx + warp_size * pass + lane_id);
    value[value_idx] = delta;
  }

  // decodes and discards values so the decoder resumes at the pass boundary at or just past
  // `skip`. the up to warp_size - 1 values decoded beyond `skip` stay resident in the rolling
  // buffer for the consumer, which resumes reading at `skip`. works for any mini-block size.
  // called by all threads in a thread block (`block`); the decode runs on warp 0 (`warp`).
  inline __device__ void skip_values(
    int skip,
    cg::thread_block const& block,
    cg::thread_block_tile<cudf::detail::warp_size, cg::thread_block> const& warp)
  {
    while (next_pass_start_idx() < static_cast<uint32_t>(skip) &&
           current_value_idx < num_encoded_values(true)) {
      // decode_next_pass only runs in warp 0, but advances decoder state everyone reads,
      // so everyone must sync around it
      block.sync();
      if (warp.meta_group_rank() == 0) { decode_next_pass(warp); }
      block.sync();
    }
  }

  // Decodes and skips values until the pass containing `skip` has been decoded, keeping a
  // running sum of the skipped values (indices below `skip`) and returning it. Values decoded
  // beyond `skip` stay resident in the rolling buffer for the consumer. Works for any
  // mini-block size. Called by all threads in warp 0 (`warp`); the result is only valid on
  // thread 0. This is intended for use only by the DELTA_LENGTH_BYTE_ARRAY decoder.
  inline __device__ size_t skip_values_and_sum(
    int skip, cg::thread_block_tile<cudf::detail::warp_size, cg::thread_block> const& warp)
  {
    using cudf::detail::warp_size;
    // DELTA_LENGTH_BYTE_ARRAY lengths are encoded as INT32 by convention (since the PLAIN encoding
    // uses 4-byte lengths).
    using delta_length_type = int32_t;
    auto const t            = warp.thread_rank();

    // initialize sum with first value, which is stored in the block header. cast to
    // `delta_length_type` to ensure the value is interpreted properly before promoting it
    // back to `size_t`.
    size_t sum = static_cast<delta_length_type>(value_at(0));

    // if only skipping one value, we're done already
    if (skip == 1) { return sum; }

    while (next_pass_start_idx() < static_cast<uint32_t>(skip) &&
           current_value_idx < num_encoded_values(true)) {
      // the pass decoded below produces indices [pass_first, pass_first + warp_size); the
      // header value at index 0 is not part of any pass and is already in `sum`
      auto const pass_first = max(next_pass_start_idx(), 1u);
      decode_next_pass(warp);

      auto const idx      = pass_first + t;
      size_t const val    = idx < static_cast<uint32_t>(skip) && idx < value_count
                              ? static_cast<delta_length_type>(value_at(idx))
                              : 0;
      auto const warp_sum = cg::reduce(warp, val, cg::plus<size_t>{});
      if (t == 0) { sum += warp_sum; }
      warp.sync();
    }

    return sum;
  }

  // decode the next warp_size-wide pass of the current mini-block into db->value, advancing to
  // the next mini-block once all of its passes have been decoded. Decoding a single pass at a
  // time keeps the rolling buffer footprint independent of the mini-block size. Should only be
  // called by a single warp `warp`. NOTE: lane 0's state updates are not synchronized on exit;
  // the caller must synchronize the warp (or block) before the next call so all lanes observe
  // them.
  // Decode the next warp_size-wide pass and hand this lane its value directly, bypassing the
  // rolling buffer. `value_idx` receives the absolute index of the returned value in the page's
  // value stream (index 0 is the header value, which is never produced by a pass).
  //
  // For the flat DELTA_BINARY path the warp that decodes a value is also the warp that stores it,
  // so the round trip through shared `value[]` is pure overhead. Returns false when the stream is
  // exhausted, in which case `out` and `value_idx` are untouched.
  inline __device__ bool decode_next_pass_value(
    cg::thread_block_tile<cudf::detail::warp_size, cg::thread_block> const& warp,
    zigzag128_t& out,
    uint32_t& value_idx,
    delta_stream_ring* ring = nullptr,
    delta_pipeline* pipe    = nullptr)
  {
    using cudf::detail::warp_size;

    if (not advance_past_first_value(warp)) { return false; }

    // must be read before the lane-0 update below advances current_value_idx / cur_pass
    value_idx = current_value_idx + cur_pass * warp_size + warp.thread_rank();
    out       = calc_mini_block_pass_value(cur_pass, warp, ring, pipe);

    if (warp.thread_rank() == 0) {
      if (++cur_pass == values_per_mb / warp_size) {
        cur_pass = 0;
        setup_next_mini_block(true);
      }
    }
    return true;
  }

  /**
   * @brief Can this page be decoded a whole block at a time?
   *
   * Lane L takes `vpl = block_size / warp_size` consecutive values. They share a bit width and a
   * body pointer only if they all fall in one mini-block, which needs `values_per_mb % vpl == 0`,
   * i.e. `warp_size % mini_block_count == 0`. Encoders in the wild use 4 mini-blocks.
   */
  inline __device__ bool block_decode_supported() const
  {
    using cudf::detail::warp_size;
    // Worst-case block body is block_size values at the widest legal field (64 bits). When the
    // ring is in use that must fit its span, so the choice stays compile-time inside decode_block;
    // pages that do not fit keep the pass-at-a-time decoder, which handles the ring dynamically.
    bool const body_fits =
      not delta_ring_enabled or
      (static_cast<int>(block_size) * sizeof(uint64_t) + 8 <= delta_ring_span_bytes);
    // Only 4 and 8 values-per-lane are instantiated; the spec's "block_size is a multiple of 128"
    // makes those the shapes that actually occur, and anything else keeps the pass-at-a-time
    // decoder rather than growing a third instantiation.
    auto const vpl = block_size / warp_size;
    return not error and mini_block_count > 0 and block_size >= warp_size and
           (warp_size % mini_block_count) == 0 and (vpl == 4 or vpl == 8) and body_fits;
  }

  /**
   * @brief Decode one whole block, handing each value to `store` as it is produced.
   *
   * A pass-at-a-time decoder spends a warp-wide inclusive scan, a sync and a header walk on only
   * 32 values, which leaves nothing for the ring's prefetch to overlap with. Taking a whole block
   * gives each lane `vpl` independent unpacks in flight and amortizes the scan over `block_size`
   * values instead of 32.
   *
   * Advances `block_start`, `last_value` and `value_idx`. Only valid when
   * block_decode_supported(); callers with a skip-resume or an unusual mini-block count must use
   * decode_next_pass_value() instead.
   *
   * @param store Called as `store(value_index, value)` for every value this block produces.
   */
  template <int VPL, typename StoreFn>
  inline __device__ void decode_block(
    cg::thread_block_tile<cudf::detail::warp_size, cg::thread_block> const& warp,
    delta_stream_ring& ring,
    delta_pipeline& pipe,
    uint32_t& value_idx,
    StoreFn store)
  {
    using cudf::detail::warp_size;
    static_assert(VPL > 0 and VPL <= delta_max_values_per_lane);

    auto const lane = static_cast<int>(warp.thread_rank());
    int const vpm   = static_cast<int>(values_per_mb);

    // Step k of the block covers values [k * warp_size, (k+1) * warp_size), so consecutive lanes
    // stay on consecutive values and the nz_idx loads and output stores keep the coalescing the
    // pass-at-a-time decoder had. Giving each lane vpl *consecutive* values instead would spread
    // the warp vpl apart and turn both into strided scatters -- measured ~9% slower than the
    // pass-at-a-time decoder on 64-bit columns, wiping out the ILP win.
    int const k_per_mb = vpm / warp_size;  // steps before the mini-block advances

    // block header (min_delta varint + one width byte per mini-block) stays on global: it is a
    // handful of bytes per block and L1 broadcasts it across the warp
    zigzag128_t min_delta = 0;
    int hdr_len           = 0;
    if (lane == 0) {
      auto cur  = block_start;
      min_delta = get_zz128(cur, block_end);
      hdr_len   = static_cast<int>(cur - block_start);
    }
    min_delta = warp.shfl(min_delta, 0);
    hdr_len   = warp.shfl(hdr_len, 0);

    auto const* const widths = block_start + hdr_len;
    auto const* const body   = widths + mini_block_count;

    // a mini-block body is values_per_mb * width bits, which only coincides with
    // warp_size * width when values_per_mb == 32
    int total_body = 0;
    for (int j = 0; j < static_cast<int>(mini_block_count); ++j) {
      total_body += vpm * widths[j] / CHAR_BIT;
    }

    // block_decode_supported() already guaranteed the worst-case body fits the ring's span, so
    // there is no runtime ring/global choice here -- keeping one would put a branch inside the
    // unrolled unpack below, which measured 7 extra registers and ~11% runtime.
    [[maybe_unused]] int body_pos = 0;
    if constexpr (delta_ring_enabled) {
      body_pos = static_cast<int>(body - ring.page_base);
      ring.reach(body_pos + total_body + 8, warp, pipe);
    }

    zigzag128_t d[VPL];

    // `fetch` is bound outside the loop so the ring/global choice never becomes a per-value
    // branch: leaving it inside costs 7 registers and, measured earlier, ~11% runtime.
    // mini-block index is lane-uniform at every step and advances by at most one, so the body
    // offset can be carried rather than re-summed.
    auto const unpack_all = [&](auto&& fetch) {
      int mb = 0, body_off = 0;
#pragma unroll
      for (int k = 0; k < VPL; ++k) {
        int const mb_k = k / k_per_mb;
        if (mb_k != mb) {
          body_off += vpm * widths[mb] / CHAR_BIT;
          mb = mb_k;
        }
        int const width   = widths[mb];
        int const slot    = k * warp_size + lane - mb * vpm;
        uint32_t const gi = value_idx + k * warp_size + lane;
        d[k]              = (gi < value_count) ? fetch(body_off, slot, width) + min_delta : 0;
      }
    };

    if constexpr (delta_ring_enabled) {
      unpack_all([&](int off, int slot, int width) {
        return ring.extract((body_pos + off) * CHAR_BIT + slot * width, width);
      });
    } else {
      unpack_all([&](int off, int slot, int width) {
        return unpack_bitpacked_field(body + off, block_end, slot * width, width);
      });
    }

    // All vpl unpacks above are independent and already issued, which is the ILP that matters:
    // they are the memory-latency side. The scans below cannot be collapsed into one per block --
    // that would require each lane to own a contiguous value range, which is exactly the layout
    // that destroys store coalescing -- so they stay one per warp_size values, as before.
    auto carry = last_value;
#pragma unroll
    for (int k = 0; k < VPL; ++k) {
      auto const incl   = cg::inclusive_scan(warp, d[k], cg::plus<int64_t>{});
      uint32_t const gi = value_idx + k * warp_size + lane;
      if (gi < value_count) { store(gi, carry + incl); }
      carry += warp.shfl(incl, warp_size - 1);
    }

    last_value  = carry;
    block_start = body + total_body;
    value_idx += block_size;
    warp.sync();
  }

  inline __device__ void decode_next_pass(
    cg::thread_block_tile<cudf::detail::warp_size, cg::thread_block> const& warp)
  {
    using cudf::detail::warp_size;

    if (not advance_past_first_value(warp)) { return; }

    // unpack one pass of deltas and save in db->value
    calc_mini_block_pass(cur_pass, warp);

    // advance within the mini-block; move to the next mini-block once all passes are decoded
    if (warp.thread_rank() == 0) {
      if (++cur_pass == values_per_mb / warp_size) {
        cur_pass = 0;
        setup_next_mini_block(true);
      }
    }
  }
};

}  // namespace cudf::io::parquet::detail
