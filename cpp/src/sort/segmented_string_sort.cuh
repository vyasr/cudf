/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "string_sort_config.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cuda/pipeline>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <utility>

// CUDA's block pipeline state is intentionally initialized cooperatively in shared memory.
#pragma nv_diag_suppress static_var_with_dynamic_init

namespace cudf::detail {
namespace segmented_string_sort {

constexpr size_type comparison_chunk_size   = 512;
constexpr size_type radix_prefix_bytes      = sizeof(std::uint64_t);
constexpr size_type maximum_string_size     = std::numeric_limits<size_type>::max();
constexpr size_type invalid_index           = -1;
constexpr size_type shuffle_block_size      = 256;
constexpr size_type strings_per_shuffle_cta = 2048;
constexpr size_type shuffle_copy_bytes      = 2048;
constexpr size_type shuffle_pipeline_stages = 2;

__device__ constexpr std::uint64_t byte_swap(std::uint64_t value)
{
  value = ((value & 0x00ff00ff00ff00ffULL) << 8) | ((value & 0xff00ff00ff00ff00ULL) >> 8);
  value = ((value & 0x0000ffff0000ffffULL) << 16) | ((value & 0xffff0000ffff0000ULL) >> 16);
  return (value << 32) | (value >> 32);
}

struct finish_counts {
  size_type segments{};
  size_type rows{};
};

struct refinement_counts {
  size_type candidate_starts{};
  size_type candidate_ends{};
  size_type continuing_runs{};
  size_type continuing_rows{};
  size_type completed_runs{};
  size_type duplicate_rows{};
};

__device__ inline bool strings_equal_after(column_device_view strings,
                                           size_type lhs,
                                           size_type rhs,
                                           size_type known_prefix_bytes);

__device__ inline bool find_containing_run(size_type position,
                                           size_type const* run_begins,
                                           size_type const* run_ends,
                                           size_type num_runs,
                                           size_type size,
                                           size_type& begin,
                                           size_type& end)
{
  if (num_runs == 0) {
    begin = 0;
    end   = size;
    return position < size;
  }
  auto low  = size_type{0};
  auto high = num_runs;
  while (low < high) {
    auto const middle = low + (high - low) / 2;
    if (run_ends[middle] <= position) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  if (low == num_runs || run_begins[low] > position) { return false; }
  begin = run_begins[low];
  end   = run_ends[low];
  return true;
}

CUDF_KERNEL void mark_tied_run_endpoints(std::uint64_t const* sorted_keys,
                                         size_type size,
                                         size_type const* previous_begins,
                                         size_type const* previous_ends,
                                         size_type previous_run_count,
                                         size_type* candidate_begins,
                                         size_type* candidate_ends,
                                         refinement_counts* counts)
{
  auto const position = cudf::detail::grid_1d::global_thread_id();
  if (position >= size) { return; }

  auto segment_begin = size_type{};
  auto segment_end   = size_type{};
  if (!find_containing_run(position,
                           previous_begins,
                           previous_ends,
                           previous_run_count,
                           size,
                           segment_begin,
                           segment_end)) {
    return;
  }

  auto const begins_tie =
    position + 1 < segment_end && sorted_keys[position] == sorted_keys[position + 1] &&
    (position == segment_begin || sorted_keys[position - 1] != sorted_keys[position]);
  auto const ends_tie =
    position > segment_begin && sorted_keys[position - 1] == sorted_keys[position] &&
    (position + 1 == segment_end || sorted_keys[position] != sorted_keys[position + 1]);
  if (begins_tie) {
    auto const slot        = atomicAdd(&counts->candidate_starts, size_type{1});
    candidate_begins[slot] = position;
  }
  if (ends_tie) {
    auto const slot      = atomicAdd(&counts->candidate_ends, size_type{1});
    candidate_ends[slot] = position + 1;
  }
}

template <typename Offset, bool eliminate_exact_duplicates>
CUDF_KERNEL void classify_tied_runs(Offset const* offsets,
                                    column_device_view strings,
                                    size_type strings_offset,
                                    size_type const* indices,
                                    size_type const* candidate_begins,
                                    size_type const* candidate_ends,
                                    size_type candidate_count,
                                    size_type byte_offset,
                                    size_type radix_run_min,
                                    bool last_pass,
                                    size_type* continuing_begins,
                                    size_type* continuing_ends,
                                    size_type* final_begins,
                                    size_type* final_ends,
                                    size_type* final_prefix_bytes,
                                    refinement_counts* refinement,
                                    finish_counts* finish)
{
  auto const run = static_cast<size_type>(blockIdx.x);
  if (run >= candidate_count) { return; }
  auto const begin = candidate_begins[run];
  auto const end   = candidate_ends[run];

  auto minimum_remaining = maximum_string_size;
  auto maximum_remaining = size_type{0};
  for (auto position = begin + static_cast<size_type>(threadIdx.x); position < end;
       position += static_cast<size_type>(blockDim.x)) {
    auto const row = indices[position];
    auto const row_size =
      static_cast<size_type>(offsets[strings_offset + row + 1] - offsets[strings_offset + row]);
    auto const remaining = row_size > byte_offset ? row_size - byte_offset : size_type{0};
    minimum_remaining    = min(minimum_remaining, remaining);
    maximum_remaining    = max(maximum_remaining, remaining);
  }
  using block_reduce = cub::BlockReduce<size_type, 256>;
  __shared__ typename block_reduce::TempStorage reduction_storage;
  minimum_remaining = block_reduce(reduction_storage).Reduce(minimum_remaining, cuda::minimum<>{});
  __syncthreads();
  maximum_remaining = block_reduce(reduction_storage).Reduce(maximum_remaining, cuda::maximum<>{});
  __shared__ size_type minimum_bytes;
  __shared__ size_type maximum_bytes;
  if (threadIdx.x == 0) {
    minimum_bytes = minimum_remaining;
    maximum_bytes = maximum_remaining;
  }
  __syncthreads();

  auto const full_chunk = minimum_bytes >= size_type{8};
  if (!full_chunk && minimum_bytes == maximum_bytes) {
    if (threadIdx.x == 0) { atomicAdd(&refinement->completed_runs, size_type{1}); }
    return;
  }

  auto const proven_prefix = full_chunk ? byte_offset + size_type{8} : byte_offset;
  if (full_chunk && !last_pass && end - begin >= radix_run_min) {
    if (threadIdx.x != 0) { return; }
    auto const slot         = atomicAdd(&refinement->continuing_runs, size_type{1});
    continuing_begins[slot] = begin;
    continuing_ends[slot]   = end;
    atomicAdd(&refinement->continuing_rows, end - begin);
    return;
  }

  if constexpr (eliminate_exact_duplicates) {
    // Only terminal unresolved runs pay for exact verification. Comparing after the proven prefix
    // avoids rereading radix bytes without confusing a short value with a zero-padded longer one.
    auto mismatch = size_type{0};
    for (auto position = begin + static_cast<size_type>(threadIdx.x); position < end;
         position += static_cast<size_type>(blockDim.x)) {
      mismatch |= !strings_equal_after(strings, indices[begin], indices[position], proven_prefix);
    }
    __syncthreads();
    mismatch = block_reduce(reduction_storage).Reduce(mismatch, cuda::maximum<>{});
    if (threadIdx.x != 0) { return; }
    if (mismatch == 0) {
      atomicAdd(&refinement->completed_runs, size_type{1});
      atomicAdd(&refinement->duplicate_rows, end - begin);
      return;
    }
  } else {
    if (threadIdx.x != 0) { return; }
  }

  auto const slot          = atomicAdd(&finish->segments, size_type{1});
  final_begins[slot]       = begin;
  final_ends[slot]         = end;
  final_prefix_bytes[slot] = proven_prefix;
  atomicAdd(&finish->rows, end - begin);
}

__device__ inline bool strings_equal_after(column_device_view strings,
                                           size_type lhs,
                                           size_type rhs,
                                           size_type known_prefix_bytes)
{
  auto const left  = strings.element<string_view>(lhs);
  auto const right = strings.element<string_view>(rhs);
  if (left.size_bytes() != right.size_bytes()) { return false; }
  for (auto byte = known_prefix_bytes; byte < left.size_bytes(); ++byte) {
    if (left.data()[byte] != right.data()[byte]) { return false; }
  }
  return true;
}

template <typename Offset>
CUDF_KERNEL __launch_bounds__(shuffle_block_size) void make_first_keys(Offset const* offsets,
                                                                       char const* chars,
                                                                       size_type size,
                                                                       std::uint64_t* keys)
{
  __shared__ __align__(16) Offset shared_offsets[strings_per_shuffle_cta + 1];
  __shared__ __align__(16) char shared_chars[shuffle_copy_bytes * shuffle_pipeline_stages];
  __shared__
    cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, shuffle_pipeline_stages>
      pipeline_state;

  auto const block       = cooperative_groups::this_thread_block();
  auto pipeline          = cuda::make_pipeline(block, &pipeline_state);
  auto const block_begin = static_cast<size_type>(blockIdx.x) * strings_per_shuffle_cta;
  auto const block_size  = min(strings_per_shuffle_cta, size - block_begin);
  for (auto index = static_cast<size_type>(threadIdx.x); index <= block_size;
       index += static_cast<size_type>(blockDim.x)) {
    shared_offsets[index] = offsets[block_begin + index];
  }
  block.sync();

  auto const chars_begin = static_cast<Offset>(shared_offsets[0] & ~Offset{15});
  auto const chars_end   = shared_offsets[block_size];
  auto const char_count  = static_cast<size_type>(chars_end - chars_begin);
  auto const iterations =
    max(size_type{1}, (char_count + shuffle_copy_bytes - 1) / shuffle_copy_bytes);
  auto next_copy = size_type{0};

  auto issue_copy = [&](size_type stage) {
    auto const copied = next_copy * shuffle_copy_bytes;
    auto const bytes  = min(shuffle_copy_bytes, max(size_type{0}, char_count - copied));
    pipeline.producer_acquire();
    cuda::memcpy_async(block,
                       shared_chars + stage * shuffle_copy_bytes,
                       chars + chars_begin + copied,
                       static_cast<std::size_t>(bytes),
                       pipeline);
    pipeline.producer_commit();
    ++next_copy;
  };
  issue_copy(0);

  auto local_row  = static_cast<size_type>(threadIdx.x);
  auto byte       = size_type{0};
  auto key        = std::uint64_t{0};
  auto skip_empty = [&] {
    while (local_row < block_size && shared_offsets[local_row] == shared_offsets[local_row + 1]) {
      keys[block_begin + local_row] = 0;
      local_row += static_cast<size_type>(blockDim.x);
    }
  };
  skip_empty();

  for (auto iteration = size_type{0}; iteration < iterations; ++iteration) {
    auto const current_stage = iteration % shuffle_pipeline_stages;
    if (next_copy < iterations) { issue_copy(next_copy % shuffle_pipeline_stages); }
    pipeline.consumer_wait();
    auto const window_begin = iteration * shuffle_copy_bytes;
    auto const window_end   = min(window_begin + shuffle_copy_bytes, char_count);

    while (local_row < block_size) {
      auto const row_begin = static_cast<size_type>(shared_offsets[local_row] - chars_begin);
      auto const row_size =
        static_cast<size_type>(shared_offsets[local_row + 1] - shared_offsets[local_row]);
      auto const key_size = min(size_type{8}, row_size);
      if (row_begin + byte >= window_end) { break; }
      while (byte < key_size && row_begin + byte < window_end) {
        key = (key << 8) |
              static_cast<unsigned char>(
                shared_chars[current_stage * shuffle_copy_bytes + row_begin + byte - window_begin]);
        ++byte;
      }
      if (byte != key_size) { break; }
      keys[block_begin + local_row] = key << (8 * (8 - key_size));
      local_row += static_cast<size_type>(blockDim.x);
      byte = 0;
      key  = 0;
      skip_empty();
    }
    pipeline.consumer_release();
  }
}

template <typename Offset, typename IndexIterator>
CUDF_KERNEL void make_subsequent_keys(Offset const* offsets,
                                      char const* chars,
                                      size_type strings_offset,
                                      IndexIterator indices,
                                      size_type const* run_begins,
                                      size_type const* run_ends,
                                      size_type num_runs,
                                      std::uint64_t* keys,
                                      size_type size,
                                      size_type byte_offset)
{
  auto const position = cudf::detail::grid_1d::global_thread_id();
  if (position >= size) { return; }
  size_type run_begin, run_end;
  if (!find_containing_run(position, run_begins, run_ends, num_runs, size, run_begin, run_end)) {
    return;
  }

  auto const row       = indices[position];
  auto const row_begin = offsets[strings_offset + row];
  auto const row_end   = offsets[strings_offset + row + 1];
  auto const remaining = row_end - row_begin > byte_offset ? row_end - row_begin - byte_offset : 0;
  auto const key_size  = min(Offset{8}, remaining);
  auto key             = std::uint64_t{0};
  if (key_size == radix_prefix_bytes) {
    auto const start         = row_begin + byte_offset;
    auto const aligned_start = start & ~Offset{radix_prefix_bytes - 1};
    auto const shift         = static_cast<unsigned int>((start - aligned_start) * 8);
    auto const* words        = reinterpret_cast<std::uint64_t const*>(chars + aligned_start);
    auto native_word         = words[0] >> shift;
    if (shift != 0) {
      // Reading a complete aligned word is safe only inside the chars allocation. A final
      // misaligned string may require a few tail-byte loads instead of the reference's overread.
      if (aligned_start + 2 * radix_prefix_bytes <= offsets[strings_offset + size]) {
        native_word |= words[1] << (64 - shift);
      } else {
        auto const loaded_bytes = radix_prefix_bytes - shift / 8;
        for (auto byte = loaded_bytes; byte < radix_prefix_bytes; ++byte) {
          native_word |= std::uint64_t{static_cast<unsigned char>(chars[start + byte])}
                         << (8 * byte);
        }
      }
    }
    key = byte_swap(native_word);
  } else {
    for (Offset byte = 0; byte < key_size; ++byte) {
      key = (key << 8) | static_cast<unsigned char>(chars[row_begin + byte_offset + byte]);
    }
    key <<= 8 * (8 - key_size);
  }
  keys[position] = key;
}

struct valid_row_predicate {
  __device__ bool operator()(size_type row) const { return not strings.is_null(row); }
  column_device_view strings;
};

struct null_row_predicate {
  __device__ bool operator()(size_type row) const { return strings.is_null(row); }
  column_device_view strings;
};

struct comparison_value {
  size_type row;
  size_type bytes;
  char const* data;
};

template <typename Comparator>
__device__ comparison_value load_comparison_value(size_type row,
                                                  size_type known_prefix_bytes,
                                                  Comparator comparator)
{
  if (row == invalid_index) { return {row, 0, nullptr}; }
  auto const value = comparator.d_column.template element<string_view>(row);
  return {row, value.size_bytes() - known_prefix_bytes, value.data() + known_prefix_bytes};
}

template <bool ascending>
__device__ bool stable_string_less(comparison_value lhs, comparison_value rhs)
{
  if (lhs.row == invalid_index) { return false; }
  if (rhs.row == invalid_index) { return true; }
  auto const comparison =
    string_view{lhs.data, lhs.bytes}.compare(string_view{rhs.data, rhs.bytes});
  if (comparison != 0) { return ascending ? comparison < 0 : comparison > 0; }
  return lhs.row < rhs.row;
}

CUDF_KERNEL void compute_finish_chunk_counts(size_type const* final_begins,
                                             size_type const* final_ends,
                                             size_type num_segments,
                                             size_type* chunk_counts)
{
  auto const segment = cudf::detail::grid_1d::global_thread_id();
  if (segment >= num_segments) { return; }
  auto const length     = final_ends[segment] - final_begins[segment];
  chunk_counts[segment] = length / comparison_chunk_size + (length % comparison_chunk_size != 0);
}

CUDF_KERNEL void compute_total_finish_chunks(size_type const* chunk_offsets,
                                             size_type const* chunk_counts,
                                             size_type num_segments,
                                             size_type* total_chunks)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *total_chunks = chunk_offsets[num_segments - 1] + chunk_counts[num_segments - 1];
  }
}

CUDF_KERNEL void expand_finish_chunks(size_type const* final_begins,
                                      size_type const* final_ends,
                                      size_type const* final_prefix_bytes,
                                      size_type const* chunk_offsets,
                                      size_type num_segments,
                                      size_type* chunk_begins,
                                      size_type* chunk_sizes,
                                      size_type* chunk_run_offsets,
                                      size_type* chunk_run_counts,
                                      size_type* chunk_prefix_bytes)
{
  auto const segment = cudf::detail::grid_1d::global_thread_id();
  if (segment >= num_segments) { return; }
  auto const run_begin  = final_begins[segment];
  auto const run_length = final_ends[segment] - run_begin;
  auto const run_offset = chunk_offsets[segment];
  auto const chunk_count =
    run_length / comparison_chunk_size + (run_length % comparison_chunk_size != 0);
  for (size_type chunk = 0; chunk < chunk_count; ++chunk) {
    auto const slot          = run_offset + chunk;
    auto const begin         = run_begin + chunk * comparison_chunk_size;
    chunk_begins[slot]       = begin;
    chunk_sizes[slot]        = min(comparison_chunk_size, final_ends[segment] - begin);
    chunk_run_offsets[slot]  = run_offset;
    chunk_run_counts[slot]   = chunk_count;
    chunk_prefix_bytes[slot] = final_prefix_bytes[segment];
  }
}

template <bool ascending, typename Comparator>
CUDF_KERNEL __launch_bounds__(comparison_chunk_size, 1) void bitonic_sort_finish_chunks(
  size_type const* input,
  size_type* sorted_chunks,
  size_type const* chunk_begins,
  size_type const* chunk_sizes,
  size_type const* chunk_prefix_bytes,
  Comparator comparator)
{
  // Cache string metadata as in the source network so comparisons do not reload column offsets.
  __shared__ comparison_value values[comparison_chunk_size];
  auto const chunk              = static_cast<size_type>(blockIdx.x);
  auto const lane               = static_cast<size_type>(threadIdx.x);
  auto const begin              = chunk_begins[chunk];
  auto const length             = chunk_sizes[chunk];
  auto const known_prefix_bytes = chunk_prefix_bytes[chunk];
  values[lane]                  = load_comparison_value(
    lane < length ? input[begin + lane] : invalid_index, known_prefix_bytes, comparator);
  __syncthreads();

  auto sort_size = comparison_chunk_size;
  if (length < comparison_chunk_size) {
    sort_size = length - 1;
    sort_size |= sort_size >> 1;
    sort_size |= sort_size >> 2;
    sort_size |= sort_size >> 4;
    sort_size |= sort_size >> 8;
    sort_size |= sort_size >> 16;
    ++sort_size;
  }
  for (size_type sequence = 2; sequence <= sort_size; sequence <<= 1) {
    for (size_type stride = sequence >> 1; stride > 0; stride >>= 1) {
      auto const peer = lane ^ stride;
      if (lane < sort_size && peer > lane) {
        auto const ascending_network = (lane & sequence) == 0;
        auto const should_swap       = ascending_network
                                         ? stable_string_less<ascending>(values[peer], values[lane])
                                         : stable_string_less<ascending>(values[lane], values[peer]);
        if (should_swap) {
          auto const temporary = values[lane];
          values[lane]         = values[peer];
          values[peer]         = temporary;
        }
      }
      __syncthreads();
    }
  }
  if (lane < length) { sorted_chunks[begin + lane] = values[lane].row; }
}

template <bool ascending, typename Comparator>
CUDF_KERNEL __launch_bounds__(comparison_chunk_size,
                              1) void merge_all_sibling_chunks(size_type* output,
                                                               size_type const* sorted_chunks,
                                                               size_type const* chunk_begins,
                                                               size_type const* chunk_sizes,
                                                               size_type const* chunk_run_offsets,
                                                               size_type const* chunk_run_counts,
                                                               size_type const* chunk_prefix_bytes,
                                                               Comparator comparator)
{
  __shared__ comparison_value sibling_values[comparison_chunk_size];
  auto const chunk              = static_cast<size_type>(blockIdx.x);
  auto const lane               = static_cast<size_type>(threadIdx.x);
  auto const begin              = chunk_begins[chunk];
  auto const length             = chunk_sizes[chunk];
  auto const run_offset         = chunk_run_offsets[chunk];
  auto const chunk_count        = chunk_run_counts[chunk];
  auto const known_prefix_bytes = chunk_prefix_bytes[chunk];
  auto const local_chunk        = chunk - run_offset;
  auto const active             = lane < length;
  auto const value              = load_comparison_value(
    active ? sorted_chunks[begin + lane] : invalid_index, known_prefix_bytes, comparator);
  auto rank = lane;

  for (size_type sibling = 0; sibling < chunk_count; ++sibling) {
    if (sibling == local_chunk) { continue; }
    __syncthreads();
    auto const sibling_slot   = run_offset + sibling;
    auto const sibling_begin  = chunk_begins[sibling_slot];
    auto const sibling_length = chunk_sizes[sibling_slot];
    sibling_values[lane]      = load_comparison_value(
      lane < sibling_length ? sorted_chunks[sibling_begin + lane] : invalid_index,
      known_prefix_bytes,
      comparator);
    __syncthreads();
    if (!active) { continue; }

    // The original row index makes the comparison a strict total order. The resulting lower-bound
    // rank is equivalent to the source's earlier-chunk upper/later-chunk lower bounds while also
    // preserving stability if radix refinement has rearranged equal rows across chunk boundaries.
    auto lower = size_type{0};
    auto upper = sibling_length;
    while (lower < upper) {
      auto const middle = lower + (upper - lower) / 2;
      if (stable_string_less<ascending>(sibling_values[middle], value)) {
        lower = middle + 1;
      } else {
        upper = middle;
      }
    }
    rank += lower;
  }
  if (active) { output[chunk_begins[run_offset] + rank] = value.row; }
}

inline void segmented_radix_sort(std::uint64_t const* keys_in,
                                 std::uint64_t* keys_out,
                                 size_type const* values_in,
                                 size_type* values_out,
                                 size_type size,
                                 size_type num_segments,
                                 size_type const* segment_begins,
                                 size_type const* segment_ends,
                                 bool ascending,
                                 rmm::device_buffer& temp_storage,
                                 cuda::stream_ref stream)
{
  std::size_t temp_storage_bytes = 0;
  auto invoke                    = [&](void* temp_storage) {
    if (ascending) {
      return cub::DeviceSegmentedRadixSort::SortPairs(temp_storage,
                                                      temp_storage_bytes,
                                                      keys_in,
                                                      keys_out,
                                                      values_in,
                                                      values_out,
                                                      size,
                                                      num_segments,
                                                      segment_begins,
                                                      segment_ends,
                                                      0,
                                                      64,
                                                      stream.get());
    }
    return cub::DeviceSegmentedRadixSort::SortPairsDescending(temp_storage,
                                                              temp_storage_bytes,
                                                              keys_in,
                                                              keys_out,
                                                              values_in,
                                                              values_out,
                                                              size,
                                                              num_segments,
                                                              segment_begins,
                                                              segment_ends,
                                                              0,
                                                              64,
                                                              stream.get());
  };
  CUDF_CUDA_TRY(invoke(nullptr));
  if (temp_storage_bytes > temp_storage.size()) {
    temp_storage =
      rmm::device_buffer(temp_storage_bytes, stream, cudf::get_current_device_resource_ref());
  }
  CUDF_CUDA_TRY(invoke(temp_storage.data()));
}

inline void global_radix_sort(std::uint64_t const* keys_in,
                              std::uint64_t* keys_out,
                              size_type const* values_in,
                              size_type* values_out,
                              size_type size,
                              bool ascending,
                              rmm::device_buffer& temp_storage,
                              cuda::stream_ref stream)
{
  std::size_t temp_storage_bytes = 0;
  auto invoke                    = [&](void* storage) {
    if (ascending) {
      return cub::DeviceRadixSort::SortPairs(storage,
                                             temp_storage_bytes,
                                             keys_in,
                                             keys_out,
                                             values_in,
                                             values_out,
                                             size,
                                             0,
                                             64,
                                             stream.get());
    }
    return cub::DeviceRadixSort::SortPairsDescending(storage,
                                                     temp_storage_bytes,
                                                     keys_in,
                                                     keys_out,
                                                     values_in,
                                                     values_out,
                                                     size,
                                                     0,
                                                     64,
                                                     stream.get());
  };
  CUDF_CUDA_TRY(invoke(nullptr));
  if (temp_storage_bytes > temp_storage.size()) {
    temp_storage =
      rmm::device_buffer(temp_storage_bytes, stream, cudf::get_current_device_resource_ref());
  }
  CUDF_CUDA_TRY(invoke(temp_storage.data()));
}

/**
 * @brief Sort string row indices using iterative big-endian prefix refinement.
 *
 * The work decomposition follows string_sort aa13a98: global prefix sorting, tied-run refinement,
 * 512-item bitonic chunks, and stable ranking against every sibling chunk. Length classification
 * protects libcudf lexical semantics when zero padding cannot establish a complete equal prefix.
 */
template <typename Comparator>
void sorted_order(column_view const& input,
                  mutable_column_view& output,
                  bool ascending,
                  null_order null_precedence,
                  Comparator comparator,
                  segmented_string_sort_config const& tuning,
                  cuda::stream_ref stream)
{
  auto const size         = input.size();
  auto const temp_mr      = cudf::get_current_device_resource_ref();
  auto const strings_view = strings_column_view{input};
  auto const offsets      = strings_view.offsets();
  auto const chars        = strings_view.chars_begin(stream);
  auto const exec         = rmm::exec_policy_nosync(stream, temp_mr);
  auto const valid_size   = size - input.null_count();
  auto const null_size    = size - valid_size;
  auto indices_a          = rmm::device_uvector<size_type>(valid_size, stream, temp_mr);
  auto indices_b          = rmm::device_uvector<size_type>(valid_size, stream, temp_mr);
  auto null_indices       = rmm::device_uvector<size_type>(null_size, stream, temp_mr);
  auto const rows         = cuda::counting_iterator<size_type>{0};
  if (null_size == 0) {
    thrust::sequence(exec, indices_a.begin(), indices_a.end(), 0);
  } else {
    thrust::copy_if(
      exec, rows, rows + size, indices_a.begin(), valid_row_predicate{comparator.d_column});
    thrust::copy_if(
      exec, rows, rows + size, null_indices.begin(), null_row_predicate{comparator.d_column});
  }

  auto const maximum_radix_passes = static_cast<size_type>(tuning.lexic_precision);
  auto synchronization_points     = size_type{0};
  auto trace_readbacks            = size_type{0};
  auto const radix_run_min        = static_cast<size_type>(tuning.radix_run_min);
  if (tuning.trace) {
    std::fprintf(stderr,
                 "segmented-string-sort bytes=%d precision=%d radix-run-min=%d "
                 "exact-duplicates=%d valid=%d nulls=%d\n",
                 radix_prefix_bytes,
                 maximum_radix_passes,
                 tuning.radix_run_min,
                 tuning.eliminate_exact_duplicates,
                 valid_size,
                 null_size);
  }

  auto keys_in  = rmm::device_uvector<std::uint64_t>(valid_size, stream, temp_mr);
  auto keys_out = rmm::device_uvector<std::uint64_t>(valid_size, stream, temp_mr);

  // Continuing runs contain at least radix_run_min rows; cutoff runs never enter these arrays.
  // Reserve one slot for the virtual first-pass run even on smaller inputs.
  auto const maximum_continuing_runs = std::max(size_type{1}, valid_size / radix_run_min);
  auto begins_a = rmm::device_uvector<size_type>(maximum_continuing_runs, stream, temp_mr);
  auto ends_a   = rmm::device_uvector<size_type>(maximum_continuing_runs, stream, temp_mr);
  auto begins_b = rmm::device_uvector<size_type>(maximum_continuing_runs, stream, temp_mr);
  auto ends_b   = rmm::device_uvector<size_type>(maximum_continuing_runs, stream, temp_mr);

  auto run_begins = rmm::device_uvector<size_type>(valid_size / 2, stream, temp_mr);
  auto run_ends   = rmm::device_uvector<size_type>(valid_size / 2, stream, temp_mr);

  // Every recorded final segment contains at least two rows, so half the row count is a tight
  // upper bound.
  auto const maximum_finish_items = valid_size / 2;
  auto final_begins       = rmm::device_uvector<size_type>(maximum_finish_items, stream, temp_mr);
  auto final_ends         = rmm::device_uvector<size_type>(maximum_finish_items, stream, temp_mr);
  auto final_prefix_bytes = rmm::device_uvector<size_type>(maximum_finish_items, stream, temp_mr);
  auto counts = cudf::detail::device_scalar<finish_counts>(finish_counts{}, stream, temp_mr);
  auto refinement =
    cudf::detail::device_scalar<refinement_counts>(refinement_counts{}, stream, temp_mr);

  thrust::fill_n(exec, begins_a.begin(), 1, size_type{0});
  thrust::fill_n(exec, ends_a.begin(), 1, valid_size);

  auto* current_indices        = indices_a.data();
  auto* other_indices          = indices_b.data();
  auto* current_begins         = begins_a.data();
  auto* current_ends           = ends_a.data();
  auto* next_begins            = begins_b.data();
  auto* next_ends              = ends_b.data();
  auto const config            = cudf::detail::grid_1d{valid_size, 256};
  auto const first_pass_config = cudf::detail::grid_1d{
    valid_size, shuffle_block_size, strings_per_shuffle_cta / shuffle_block_size};
  auto cub_temp_storage = rmm::device_buffer{};
  size_type num_segments{1};
  // This counts algorithm-visible workspace allocations rather than upstream pool growth. It makes
  // stage-to-stage buffer lifetime changes observable without replacing the caller's memory
  // resource.
  auto allocation_count = size_type{15 + (null_indices.size() > 0 ? 1 : 0)};
  auto active_rows      = valid_size;

  for (size_type pass = 0; pass < maximum_radix_passes && num_segments > 0; ++pass) {
    auto const launch_keys = [&]<typename Offset>() {
      auto const* typed_offsets = offsets.head<Offset>() + input.offset();
      if (pass == 0 && null_size == 0) {
        make_first_keys<Offset><<<first_pass_config.num_blocks,
                                  first_pass_config.num_threads_per_block,
                                  0,
                                  stream.get()>>>(typed_offsets, chars, valid_size, keys_in.data());
      } else {
        make_subsequent_keys<Offset>
          <<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
            offsets.head<Offset>(),
            chars,
            input.offset(),
            current_indices,
            current_begins,
            current_ends,
            pass == 0 ? size_type{0} : num_segments,
            keys_in.data(),
            valid_size,
            pass * radix_prefix_bytes);
      }
    };
    if (offsets.type().id() == type_id::INT64) {
      launch_keys.template operator()<int64_t>();
    } else {
      launch_keys.template operator()<size_type>();
    }
    CUDF_CUDA_TRY(cudaGetLastError());
    auto const cub_storage_was_empty = cub_temp_storage.size() == 0;
    if (pass == 0) {
      global_radix_sort(keys_in.data(),
                        keys_out.data(),
                        current_indices,
                        other_indices,
                        valid_size,
                        ascending,
                        cub_temp_storage,
                        stream);
    } else {
      CUDF_CUDA_TRY(cudf::detail::memcpy_async(
        other_indices, current_indices, sizeof(size_type) * valid_size, stream));
      segmented_radix_sort(keys_in.data(),
                           keys_out.data(),
                           current_indices,
                           other_indices,
                           valid_size,
                           num_segments,
                           current_begins,
                           current_ends,
                           ascending,
                           cub_temp_storage,
                           stream);
    }
    if (cub_storage_was_empty && cub_temp_storage.size() > 0) { ++allocation_count; }
    std::swap(current_indices, other_indices);

    refinement.set_value_async(refinement_counts{}, stream);
    auto const previous_run_count = pass == 0 ? size_type{0} : num_segments;
    mark_tied_run_endpoints<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
      keys_out.data(),
      valid_size,
      current_begins,
      current_ends,
      previous_run_count,
      run_begins.data(),
      run_ends.data(),
      refinement.data());
    CUDF_CUDA_TRY(cudaGetLastError());

    auto observed = refinement.value(stream);
    ++synchronization_points;
    CUDF_EXPECTS(observed.candidate_starts == observed.candidate_ends,
                 "Segmented string sort produced mismatched run endpoints");
    auto const candidate_count = observed.candidate_starts;
    if (candidate_count > 0) {
      thrust::sort(exec, run_begins.begin(), run_begins.begin() + candidate_count);
      thrust::sort(exec, run_ends.begin(), run_ends.begin() + candidate_count);
      auto const last_pass             = pass + 1 == maximum_radix_passes;
      auto const launch_classification = [&]<typename Offset, bool eliminate_duplicates>() {
        classify_tied_runs<Offset, eliminate_duplicates>
          <<<candidate_count, 256, 0, stream.get()>>>(offsets.head<Offset>(),
                                                      comparator.d_column,
                                                      input.offset(),
                                                      current_indices,
                                                      run_begins.data(),
                                                      run_ends.data(),
                                                      candidate_count,
                                                      pass * radix_prefix_bytes,
                                                      radix_run_min,
                                                      last_pass,
                                                      next_begins,
                                                      next_ends,
                                                      final_begins.data(),
                                                      final_ends.data(),
                                                      final_prefix_bytes.data(),
                                                      refinement.data(),
                                                      counts.data());
      };
      auto const dispatch_classification = [&]<typename Offset>() {
        if (tuning.eliminate_exact_duplicates) {
          launch_classification.template operator()<Offset, true>();
        } else {
          launch_classification.template operator()<Offset, false>();
        }
      };
      if (offsets.type().id() == type_id::INT64) {
        dispatch_classification.template operator()<int64_t>();
      } else {
        dispatch_classification.template operator()<size_type>();
      }
      CUDF_CUDA_TRY(cudaGetLastError());
    }

    observed = refinement.value(stream);
    ++synchronization_points;
    num_segments = observed.continuing_runs;
    if (num_segments > 0) {
      thrust::sort_by_key(exec, next_begins, next_begins + num_segments, next_ends);
    }
    if (tuning.trace) {
      std::fprintf(stderr,
                   "segmented-string-sort pass=%d active-rows=%d tied-runs=%d "
                   "continuing-runs=%d continuing-rows=%d completed-runs=%d duplicate-rows=%d\n",
                   pass,
                   active_rows,
                   candidate_count,
                   observed.continuing_runs,
                   observed.continuing_rows,
                   observed.completed_runs,
                   observed.duplicate_rows);
    }

    std::swap(current_begins, next_begins);
    std::swap(current_ends, next_ends);
    active_rows = observed.continuing_rows;
  }

  // Transfer the finish launch metadata together after refinement.
  auto const finish = counts.value(stream);
  ++synchronization_points;
  auto total_chunks = size_type{0};
  if (finish.segments > 0) {
    // Refinement endpoints are dead here and have enough capacity for one count/offset per finish
    // run, so the scan needs no additional row-sized storage.
    auto* chunk_counts      = run_begins.data();
    auto* chunk_offsets     = run_ends.data();
    auto const chunk_config = cudf::detail::grid_1d{finish.segments, 256};
    compute_finish_chunk_counts<<<chunk_config.num_blocks,
                                  chunk_config.num_threads_per_block,
                                  0,
                                  stream.get()>>>(
      final_begins.data(), final_ends.data(), finish.segments, chunk_counts);
    std::size_t scan_bytes = 0;
    CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
      nullptr, scan_bytes, chunk_counts, chunk_offsets, finish.segments, stream.get()));
    if (scan_bytes > cub_temp_storage.size()) {
      cub_temp_storage = rmm::device_buffer(scan_bytes, stream, temp_mr);
      ++allocation_count;
    }
    CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(cub_temp_storage.data(),
                                                scan_bytes,
                                                chunk_counts,
                                                chunk_offsets,
                                                finish.segments,
                                                stream.get()));
    auto device_total_chunks =
      cudf::detail::device_scalar<size_type>(size_type{0}, stream, temp_mr);
    compute_total_finish_chunks<<<1, 1, 0, stream.get()>>>(
      chunk_offsets, chunk_counts, finish.segments, device_total_chunks.data());
    total_chunks = device_total_chunks.value(stream);
    ++synchronization_points;

    auto chunk_begins       = rmm::device_uvector<size_type>(total_chunks, stream, temp_mr);
    auto chunk_sizes        = rmm::device_uvector<size_type>(total_chunks, stream, temp_mr);
    auto chunk_run_offsets  = rmm::device_uvector<size_type>(total_chunks, stream, temp_mr);
    auto chunk_run_counts   = rmm::device_uvector<size_type>(total_chunks, stream, temp_mr);
    auto chunk_prefix_bytes = rmm::device_uvector<size_type>(total_chunks, stream, temp_mr);
    allocation_count += 6;
    expand_finish_chunks<<<chunk_config.num_blocks,
                           chunk_config.num_threads_per_block,
                           0,
                           stream.get()>>>(final_begins.data(),
                                           final_ends.data(),
                                           final_prefix_bytes.data(),
                                           chunk_offsets,
                                           finish.segments,
                                           chunk_begins.data(),
                                           chunk_sizes.data(),
                                           chunk_run_offsets.data(),
                                           chunk_run_counts.data(),
                                           chunk_prefix_bytes.data());
    auto const launch_finish = [&]<bool sort_ascending>() {
      bitonic_sort_finish_chunks<sort_ascending>
        <<<total_chunks, comparison_chunk_size, 0, stream.get()>>>(current_indices,
                                                                   other_indices,
                                                                   chunk_begins.data(),
                                                                   chunk_sizes.data(),
                                                                   chunk_prefix_bytes.data(),
                                                                   comparator);
      merge_all_sibling_chunks<sort_ascending>
        <<<total_chunks, comparison_chunk_size, 0, stream.get()>>>(current_indices,
                                                                   other_indices,
                                                                   chunk_begins.data(),
                                                                   chunk_sizes.data(),
                                                                   chunk_run_offsets.data(),
                                                                   chunk_run_counts.data(),
                                                                   chunk_prefix_bytes.data(),
                                                                   comparator);
    };
    if (ascending) {
      launch_finish.template operator()<true>();
    } else {
      launch_finish.template operator()<false>();
    }
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  if (tuning.trace) {
    std::fprintf(stderr,
                 "segmented-string-sort finish-segments=%d finish-rows=%d chunks=%d "
                 "allocations=%d sync-points=%d trace-readbacks=%d\n",
                 finish.segments,
                 finish.rows,
                 total_chunks,
                 allocation_count,
                 synchronization_points,
                 trace_readbacks);
  }

  {
    auto const nulls_first  = ascending == (null_precedence == null_order::BEFORE);
    auto const valid_offset = nulls_first ? null_size : size_type{0};
    auto const null_offset  = nulls_first ? size_type{0} : valid_size;
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(output.begin<size_type>() + valid_offset,
                                             current_indices,
                                             sizeof(size_type) * valid_size,
                                             stream));
    if (null_size > 0) {
      CUDF_CUDA_TRY(cudf::detail::memcpy_async(output.begin<size_type>() + null_offset,
                                               null_indices.data(),
                                               sizeof(size_type) * null_size,
                                               stream));
    }
  }
}

}  // namespace segmented_string_sort
}  // namespace cudf::detail
