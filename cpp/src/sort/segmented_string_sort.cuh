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

constexpr size_type block_sort_size         = 256;
constexpr size_type comparison_chunk_size   = 512;
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

template <int bytes_per_pass>
struct radix_key_layout {
  static constexpr bool uses_full_width = bytes_per_pass == static_cast<int>(sizeof(std::uint64_t));
  static_assert(bytes_per_pass == 6 || uses_full_width,
                "Segmented string sort supports only six- and eight-byte radix keys");
  static constexpr bool stores_metadata = not uses_full_width;
};

struct finish_counts {
  size_type segments{};
  size_type rows{};
  // Retained until the obsolete derivative finishing kernels are removed in the cleanup stage.
  size_type block_tasks{};
  size_type maximum_segment_size{};
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

CUDF_KERNEL void mark_active_segments(size_type const* begins,
                                      size_type const* ends,
                                      size_type num_runs,
                                      std::uint8_t* active)
{
  auto const run = static_cast<size_type>(blockIdx.x);
  if (run >= num_runs) { return; }
  for (auto position = begins[run] + static_cast<size_type>(threadIdx.x); position < ends[run];
       position += static_cast<size_type>(blockDim.x)) {
    active[position] = 1;
  }
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

template <int bytes_per_pass, typename IndexIterator>
CUDF_KERNEL void make_keys(column_device_view strings,
                           IndexIterator indices,
                           std::uint8_t const* active,
                           std::uint64_t* keys,
                           size_type size,
                           size_type byte_offset,
                           null_order null_precedence)
{
  using key_layout    = radix_key_layout<bytes_per_pass>;
  auto const position = cudf::detail::grid_1d::global_thread_id();
  if (position >= size || active[position] == 0) { return; }

  auto const row = indices[position];
  if constexpr (key_layout::stores_metadata) {
    // Category zero sorts before category one, while category two sorts after it. Descending radix
    // sort reverses all three categories, matching libcudf's existing null-order semantics.
    if (strings.is_null(row)) {
      auto const category =
        null_precedence == null_order::BEFORE ? std::uint64_t{0} : std::uint64_t{2};
      keys[position] = category << 62;
      return;
    }
  }

  auto const value     = strings.element<string_view>(row);
  auto const remaining = value.size_bytes() > byte_offset ? value.size_bytes() - byte_offset : 0;
  auto const length    = remaining < bytes_per_pass ? remaining : bytes_per_pass;
  std::uint64_t bytes  = 0;
  for (size_type i = 0; i < bytes_per_pass; ++i) {
    bytes <<= 8;
    if (i < length) {
      // string_view comparison uses unsigned bytes, including for malformed UTF-8 and embedded NUL.
      bytes |= static_cast<unsigned char>(value.data()[byte_offset + i]);
    }
  }
  if constexpr (key_layout::stores_metadata) {
    keys[position] = (std::uint64_t{1} << 62) | (bytes << 3) | length;
  } else {
    keys[position] = bytes;
  }
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
                                      std::uint8_t const* active,
                                      std::uint64_t* keys,
                                      size_type size,
                                      size_type byte_offset)
{
  auto const position = cudf::detail::grid_1d::global_thread_id();
  if (position >= size || active[position] == 0) { return; }

  auto const row       = indices[position];
  auto const row_begin = offsets[strings_offset + row];
  auto const row_end   = offsets[strings_offset + row + 1];
  auto const remaining = row_end - row_begin > byte_offset ? row_end - row_begin - byte_offset : 0;
  auto const key_size  = min(Offset{8}, remaining);
  auto key             = std::uint64_t{0};
  if (key_size == 8) {
    std::uint64_t native_word;
    memcpy(&native_word, chars + row_begin + byte_offset, sizeof(native_word));
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

struct string_length_fn {
  __device__ size_type operator()(size_type row) const
  {
    return strings.is_null(row) ? size_type{0} : strings.element<string_view>(row).size_bytes();
  }
  column_device_view strings;
};

CUDF_KERNEL void mark_segment_endpoints(size_type const* segment_begins,
                                        size_type const* segment_ends,
                                        size_type num_slots,
                                        std::uint8_t* segment_starts,
                                        std::uint8_t* segment_ends_at)
{
  auto const slot = cudf::detail::grid_1d::global_thread_id();
  if (slot >= num_slots) { return; }
  auto const begin = segment_begins[slot];
  auto const end   = segment_ends[slot];
  if (end > begin) {
    segment_starts[begin]    = 1;
    segment_ends_at[end - 1] = 1;
  }
}

CUDF_KERNEL void mark_run_boundaries(std::uint8_t const* active,
                                     std::uint8_t const* segment_starts,
                                     std::uint8_t const* segment_ends_at,
                                     std::uint64_t const* keys,
                                     size_type size,
                                     size_type* run_starts_at,
                                     std::uint8_t* run_ends_at)
{
  auto const position = cudf::detail::grid_1d::global_thread_id();
  if (position >= size || active[position] == 0) { return; }

  auto const is_start = segment_starts[position] != 0 || position == 0 ||
                        active[position - 1] == 0 || keys[position - 1] != keys[position];
  auto const is_end = segment_ends_at[position] != 0 || position + 1 == size ||
                      active[position + 1] == 0 || keys[position + 1] != keys[position];
  run_starts_at[position] = is_start ? 1 : 0;
  run_ends_at[position]   = is_end ? 1 : 0;
}

template <int bytes_per_pass, typename IndexIterator>
CUDF_KERNEL void scatter_run_endpoints(column_device_view strings,
                                       IndexIterator indices,
                                       std::uint8_t const* active,
                                       size_type const* run_starts_at,
                                       std::uint8_t const* run_ends_at,
                                       size_type const* inclusive_run_ids,
                                       size_type size,
                                       size_type byte_offset,
                                       size_type* run_begins,
                                       size_type* run_ends,
                                       size_type* run_byte_state)
{
  using key_layout    = radix_key_layout<bytes_per_pass>;
  auto const position = cudf::detail::grid_1d::global_thread_id();
  if (position >= size || active[position] == 0) { return; }
  auto const run = inclusive_run_ids[position] - 1;
  if (run_starts_at[position] != 0) { run_begins[run] = position; }
  if (run_ends_at[position] != 0) { run_ends[run] = position + 1; }
  if constexpr (key_layout::uses_full_width) {
    auto const value_size  = strings.element<string_view>(indices[position]).size_bytes();
    auto const remaining   = value_size > byte_offset ? value_size - byte_offset : size_type{0};
    auto const valid_bytes = remaining < bytes_per_pass ? remaining : bytes_per_pass;
    auto const pass        = byte_offset / bytes_per_pass;
    // The pass generation makes stale slots smaller than every write in the current pass, avoiding
    // a full-buffer reset before this reduction.
    auto const state = ((pass + 1) << 8) | (bytes_per_pass - valid_bytes);
    atomicMax(run_byte_state + run, state);
  }
}

template <int bytes_per_pass>
__device__ size_type proven_prefix_bytes(size_type run,
                                         size_type byte_offset,
                                         size_type const* run_byte_state)
{
  using key_layout = radix_key_layout<bytes_per_pass>;
  if constexpr (key_layout::stores_metadata) {
    return byte_offset + bytes_per_pass;
  } else {
    return (run_byte_state[run] & 0xff) == 0 ? byte_offset + bytes_per_pass : byte_offset;
  }
}

template <int bytes_per_pass, typename IndexIterator>
CUDF_KERNEL void mark_nonduplicate_runs(column_device_view strings,
                                        IndexIterator indices,
                                        std::uint8_t const* active,
                                        size_type const* inclusive_run_ids,
                                        size_type const* run_begins,
                                        size_type size,
                                        size_type byte_offset,
                                        size_type const* run_byte_state,
                                        size_type* run_has_mismatch)
{
  auto const position = cudf::detail::grid_1d::global_thread_id();
  if (position >= size || active[position] == 0) { return; }

  auto const run   = inclusive_run_ids[position] - 1;
  auto const begin = run_begins[run];
  if (position == begin) { return; }

  auto const row        = indices[position];
  auto const first_row  = indices[begin];
  auto const row_null   = strings.is_null(row);
  auto const first_null = strings.is_null(first_row);
  auto const known_prefix_bytes =
    proven_prefix_bytes<bytes_per_pass>(run, byte_offset, run_byte_state);
  auto const equal = row_null == first_null &&
                     (row_null || strings_equal_after(strings, row, first_row, known_prefix_bytes));
  if (!equal) { atomicExch(run_has_mismatch + run, size_type{1}); }
}

struct rle_metrics {
  unsigned long long covered_rows{};
  unsigned long long sampled_pairs{};
  unsigned long long equal_pairs{};
};

template <int bytes_per_pass, typename IndexIterator>
CUDF_KERNEL void collect_rle_metrics(column_device_view strings,
                                     IndexIterator indices,
                                     std::uint8_t const* active,
                                     size_type const* inclusive_run_ids,
                                     size_type const* run_begins,
                                     size_type const* run_ends,
                                     size_type size,
                                     size_type minimum_run_length,
                                     size_type sampling_stride,
                                     size_type byte_offset,
                                     size_type const* run_byte_state,
                                     rle_metrics* metrics)
{
  auto const position = cudf::detail::grid_1d::global_thread_id();
  if (position >= size || active[position] == 0) { return; }
  auto const run    = inclusive_run_ids[position] - 1;
  auto const begin  = run_begins[run];
  auto const end    = run_ends[run];
  auto const length = end - begin;
  if (length < minimum_run_length) { return; }
  if (position == begin) {
    atomicAdd(&metrics->covered_rows, static_cast<unsigned long long>(length));
  }
  if (position == begin || position % sampling_stride != 0) { return; }

  auto const lhs      = indices[position - 1];
  auto const rhs      = indices[position];
  auto const lhs_null = strings.is_null(lhs);
  auto const rhs_null = strings.is_null(rhs);
  auto const known_prefix_bytes =
    proven_prefix_bytes<bytes_per_pass>(run, byte_offset, run_byte_state);
  auto const equal = lhs_null == rhs_null &&
                     (lhs_null || strings_equal_after(strings, lhs, rhs, known_prefix_bytes));
  atomicAdd(&metrics->sampled_pairs, 1ULL);
  if (equal) { atomicAdd(&metrics->equal_pairs, 1ULL); }
}

template <int bytes_per_pass, typename IndexIterator>
CUDF_KERNEL void classify_runs(column_device_view strings,
                               IndexIterator indices,
                               std::uint64_t const* sorted_keys,
                               size_type const* run_begins,
                               size_type const* run_ends,
                               size_type const* run_byte_state,
                               size_type num_slots,
                               size_type byte_offset,
                               bool last_pass,
                               bool detect_duplicates,
                               size_type const* run_has_mismatch,
                               size_type comparison_threshold,
                               size_type* next_run_flags,
                               size_type* final_begins,
                               size_type* final_ends,
                               size_type* final_prefix_bytes,
                               size_type* task_begins,
                               size_type* task_ends,
                               finish_counts* counts)
{
  using key_layout = radix_key_layout<bytes_per_pass>;
  auto const run   = cudf::detail::grid_1d::global_thread_id();
  if (run >= num_slots) { return; }
  auto const begin = run_begins[run];
  auto const end   = run_ends[run];
  if (begin == invalid_index || end - begin <= 1) { return; }
  auto const exact_duplicate_run = detect_duplicates && last_pass && run_has_mismatch[run] == 0;
  next_run_flags[run]            = 0;

  auto const row       = indices[begin];
  auto const is_null   = strings.is_null(row);
  auto const key_bytes = key_layout::stores_metadata
                           ? static_cast<size_type>(sorted_keys[begin] & 0x7)
                           : bytes_per_pass - (run_byte_state[run] & 0xff);
  // Metadata-bearing keys encode the valid-byte count, so a tied short chunk proves exact
  // equality. Full-width keys use all bits for payload, and zero padding can collide with a longer
  // string; such runs must be finished by comparison from the preceding known prefix.
  if (is_null || (key_layout::stores_metadata && key_bytes != bytes_per_pass)) { return; }
  auto const terminal_collision = key_layout::uses_full_width && key_bytes != bytes_per_pass;

  // The final radix chunk cannot distinguish long exact duplicates from strings that differ
  // later. Exact duplicate runs are already stable and need no comparison-sort finish.
  if (exact_duplicate_run) { return; }

  if (!terminal_collision && !last_pass && end - begin > comparison_threshold) {
    next_run_flags[run] = 1;
    return;
  }

  auto const final_slot = atomicAdd(&counts->segments, size_type{1});
  atomicAdd(&counts->rows, end - begin);
  final_begins[final_slot] = begin;
  final_ends[final_slot]   = end;
  if (final_prefix_bytes != nullptr) {
    final_prefix_bytes[final_slot] =
      proven_prefix_bytes<bytes_per_pass>(run, byte_offset, run_byte_state);
  }
  if (end - begin > comparison_threshold) {
    atomicMax(&counts->maximum_segment_size, end - begin);
    auto task_begin = begin;
    while (task_begin < end) {
      auto const task_slot   = atomicAdd(&counts->block_tasks, size_type{1});
      task_begins[task_slot] = task_begin;
      auto const remaining   = end - task_begin;
      auto const task_end =
        remaining > block_sort_size
          ? static_cast<size_type>(static_cast<std::int64_t>(task_begin) + block_sort_size)
          : end;
      task_ends[task_slot] = task_end;
      task_begin           = task_end;
    }
  }
}

CUDF_KERNEL void compact_next_segments(size_type const* run_begins,
                                       size_type const* run_ends,
                                       size_type const* next_run_flags,
                                       size_type const* inclusive_next_ids,
                                       size_type num_slots,
                                       size_type* next_begins,
                                       size_type* next_ends)
{
  auto const run = cudf::detail::grid_1d::global_thread_id();
  if (run >= num_slots || next_run_flags[run] == 0) { return; }
  auto const slot   = inclusive_next_ids[run] - 1;
  next_begins[slot] = run_begins[run];
  next_ends[slot]   = run_ends[run];
}

CUDF_KERNEL void update_active_runs(std::uint8_t const* active,
                                    size_type const* inclusive_run_ids,
                                    size_type const* next_run_flags,
                                    size_type size,
                                    std::uint8_t* next_active)
{
  auto const position = cudf::detail::grid_1d::global_thread_id();
  if (position >= size) { return; }
  next_active[position] =
    active[position] != 0 && next_run_flags[inclusive_run_ids[position] - 1] != 0 ? 1 : 0;
}

CUDF_KERNEL void map_final_segments(size_type const* final_begins,
                                    size_type const* final_ends,
                                    size_type const* final_prefix_bytes,
                                    size_type num_segments,
                                    size_type const* indices,
                                    size_type* final_begin_for_position,
                                    size_type* final_end_for_position,
                                    size_type* known_prefix_by_row)
{
  auto const segment = static_cast<size_type>(blockIdx.x);
  if (segment >= num_segments) { return; }
  auto const begin  = final_begins[segment];
  auto const end    = final_ends[segment];
  auto const length = static_cast<std::int64_t>(end) - begin;
  for (auto offset = static_cast<std::int64_t>(threadIdx.x); offset < length;
       offset += blockDim.x) {
    auto const position = static_cast<size_type>(static_cast<std::int64_t>(begin) + offset);
    final_begin_for_position[position] = begin;
    final_end_for_position[position]   = end;
    if (known_prefix_by_row != nullptr) {
      known_prefix_by_row[indices[position]] = final_prefix_bytes[segment];
    }
  }
}

template <typename Comparator>
__device__ bool stable_less(size_type lhs, size_type rhs, Comparator comparator)
{
  if (lhs == invalid_index) { return false; }
  if (rhs == invalid_index) { return true; }
  if (comparator(lhs, rhs)) { return true; }
  if (comparator(rhs, lhs)) { return false; }
  return lhs < rhs;
}

template <typename Comparator>
__device__ bool stable_string_less(size_type lhs,
                                   size_type rhs,
                                   size_type known_prefix_bytes,
                                   Comparator comparator)
{
  if (lhs == invalid_index) { return false; }
  if (rhs == invalid_index) { return true; }
  auto const left_value  = comparator.d_column.template element<string_view>(lhs);
  auto const right_value = comparator.d_column.template element<string_view>(rhs);
  auto const left        = string_view{left_value.data() + known_prefix_bytes,
                                left_value.size_bytes() - known_prefix_bytes};
  auto const right       = string_view{right_value.data() + known_prefix_bytes,
                                 right_value.size_bytes() - known_prefix_bytes};
  auto const comparison  = left.compare(right);
  if (comparison != 0) { return comparator.ascending ? comparison < 0 : comparison > 0; }
  return lhs < rhs;
}

CUDF_KERNEL void compute_finish_chunk_counts(size_type const* final_begins,
                                             size_type const* final_ends,
                                             size_type num_segments,
                                             size_type* chunk_counts)
{
  auto const segment = cudf::detail::grid_1d::global_thread_id();
  if (segment >= num_segments) { return; }
  auto const length     = final_ends[segment] - final_begins[segment];
  chunk_counts[segment] = (length + comparison_chunk_size - 1) / comparison_chunk_size;
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
  auto const run_begin   = final_begins[segment];
  auto const run_length  = final_ends[segment] - run_begin;
  auto const run_offset  = chunk_offsets[segment];
  auto const chunk_count = (run_length + comparison_chunk_size - 1) / comparison_chunk_size;
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

CUDF_KERNEL void map_final_prefixes(size_type const* final_begins,
                                    size_type const* final_ends,
                                    size_type const* final_prefix_bytes,
                                    size_type num_segments,
                                    size_type const* indices,
                                    size_type* known_prefix_by_row)
{
  auto const segment = static_cast<size_type>(blockIdx.x);
  if (segment >= num_segments) { return; }
  for (auto position = final_begins[segment] + static_cast<size_type>(threadIdx.x);
       position < final_ends[segment];
       position += static_cast<size_type>(blockDim.x)) {
    known_prefix_by_row[indices[position]] = final_prefix_bytes[segment];
  }
}

template <typename Comparator>
CUDF_KERNEL __launch_bounds__(comparison_chunk_size, 1) void bitonic_sort_finish_chunks(
  size_type const* input,
  size_type* sorted_chunks,
  size_type const* chunk_begins,
  size_type const* chunk_sizes,
  size_type const* chunk_prefix_bytes,
  Comparator comparator)
{
  __shared__ size_type values[comparison_chunk_size];
  auto const chunk              = static_cast<size_type>(blockIdx.x);
  auto const lane               = static_cast<size_type>(threadIdx.x);
  auto const begin              = chunk_begins[chunk];
  auto const length             = chunk_sizes[chunk];
  auto const known_prefix_bytes = chunk_prefix_bytes[chunk];
  values[lane]                  = lane < length ? input[begin + lane] : invalid_index;
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
        auto const should_swap =
          ascending_network
            ? stable_string_less(values[peer], values[lane], known_prefix_bytes, comparator)
            : stable_string_less(values[lane], values[peer], known_prefix_bytes, comparator);
        if (should_swap) {
          auto const temporary = values[lane];
          values[lane]         = values[peer];
          values[peer]         = temporary;
        }
      }
      __syncthreads();
    }
  }
  if (lane < length) { sorted_chunks[begin + lane] = values[lane]; }
}

template <typename Comparator>
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
  __shared__ size_type sibling_values[comparison_chunk_size];
  auto const chunk              = static_cast<size_type>(blockIdx.x);
  auto const lane               = static_cast<size_type>(threadIdx.x);
  auto const begin              = chunk_begins[chunk];
  auto const length             = chunk_sizes[chunk];
  auto const run_offset         = chunk_run_offsets[chunk];
  auto const chunk_count        = chunk_run_counts[chunk];
  auto const known_prefix_bytes = chunk_prefix_bytes[chunk];
  auto const local_chunk        = chunk - run_offset;
  auto const active             = lane < length;
  auto const value              = active ? sorted_chunks[begin + lane] : invalid_index;
  auto rank                     = lane;

  for (size_type sibling = 0; sibling < chunk_count; ++sibling) {
    if (sibling == local_chunk) { continue; }
    __syncthreads();
    auto const sibling_slot   = run_offset + sibling;
    auto const sibling_begin  = chunk_begins[sibling_slot];
    auto const sibling_length = chunk_sizes[sibling_slot];
    sibling_values[lane] =
      lane < sibling_length ? sorted_chunks[sibling_begin + lane] : invalid_index;
    __syncthreads();
    if (!active) { continue; }

    // The original row index makes the comparison a strict total order. The resulting lower-bound
    // rank is equivalent to the source's earlier-chunk upper/later-chunk lower bounds while also
    // preserving stability if radix refinement has rearranged equal rows across chunk boundaries.
    auto lower = size_type{0};
    auto upper = sibling_length;
    while (lower < upper) {
      auto const middle = lower + (upper - lower) / 2;
      if (stable_string_less(sibling_values[middle], value, known_prefix_bytes, comparator)) {
        lower = middle + 1;
      } else {
        upper = middle;
      }
    }
    rank += lower;
  }
  if (active) { output[chunk_begins[run_offset] + rank] = value; }
}

template <typename Comparator>
CUDF_KERNEL void finish_small_segments(size_type* indices,
                                       size_type const* final_begins,
                                       size_type const* final_ends,
                                       size_type num_segments,
                                       size_type comparison_threshold,
                                       Comparator comparator)
{
  auto const segment = cudf::detail::grid_1d::global_thread_id();
  if (segment >= num_segments) { return; }
  auto const begin = final_begins[segment];
  auto const end   = final_ends[segment];
  if (end - begin > comparison_threshold) { return; }

  // The configured threshold bounds this serial path. Tie-breaking by original row index makes the
  // result stable while remaining valid for the unstable API.
  for (auto i = begin + 1; i < end; ++i) {
    auto const value = indices[i];
    auto j           = i;
    while (j > begin && stable_less(value, indices[j - 1], comparator)) {
      indices[j] = indices[j - 1];
      --j;
    }
    indices[j] = value;
  }
}

template <typename Comparator>
CUDF_KERNEL void block_sort_tasks(size_type* indices,
                                  size_type const* task_begins,
                                  size_type const* task_ends,
                                  size_type num_tasks,
                                  Comparator comparator)
{
  auto const task = static_cast<size_type>(blockIdx.x);
  if (task >= num_tasks) { return; }
  auto const begin = task_begins[task];
  auto const end   = task_ends[task];
  __shared__ size_type values[block_sort_size];
  auto const lane        = static_cast<size_type>(threadIdx.x);
  auto const task_length = end - begin;
  auto const position    = static_cast<std::int64_t>(begin) + lane;
  values[lane]           = lane < task_length ? indices[position] : invalid_index;
  __syncthreads();

  // A 256-element bitonic sorting network. Invalid lanes compare greater than all real rows and
  // are discarded, allowing the final task of each segment to be shorter than a full block.
  for (size_type sequence = 2; sequence <= block_sort_size; sequence <<= 1) {
    for (size_type stride = sequence >> 1; stride > 0; stride >>= 1) {
      auto const peer = lane ^ stride;
      if (peer > lane) {
        auto const ascending_network = (lane & sequence) == 0;
        auto const swap_values       = ascending_network
                                         ? stable_less(values[peer], values[lane], comparator)
                                         : stable_less(values[lane], values[peer], comparator);
        if (swap_values) {
          auto const tmp = values[lane];
          values[lane]   = values[peer];
          values[peer]   = tmp;
        }
      }
      __syncthreads();
    }
  }
  if (lane < task_length) { indices[position] = values[lane]; }
}

template <typename Comparator>
CUDF_KERNEL void merge_sorted_blocks(size_type const* input,
                                     size_type* output,
                                     size_type const* final_begin_for_position,
                                     size_type const* final_end_for_position,
                                     size_type size,
                                     std::int64_t run_width,
                                     size_type comparison_threshold,
                                     Comparator comparator)
{
  auto const position = cudf::detail::grid_1d::global_thread_id();
  if (position >= size) { return; }
  auto const segment_begin = final_begin_for_position[position];
  auto const segment_end   = final_end_for_position[position];
  if (segment_begin == invalid_index || segment_end - segment_begin <= comparison_threshold) {
    output[position] = input[position];
    return;
  }

  auto const relative = static_cast<std::int64_t>(position - segment_begin);
  auto const pair_begin =
    static_cast<std::int64_t>(segment_begin) + (relative / (2 * run_width)) * (2 * run_width);
  auto const segment_end_64 = static_cast<std::int64_t>(segment_end);
  auto const middle =
    pair_begin + run_width < segment_end_64 ? pair_begin + run_width : segment_end_64;
  auto const pair_end =
    pair_begin + 2 * run_width < segment_end_64 ? pair_begin + 2 * run_width : segment_end_64;
  if (middle == pair_end) {
    output[position] = input[position];
    return;
  }

  auto const in_left     = position < middle;
  auto const other_begin = in_left ? middle : pair_begin;
  auto const other_end   = in_left ? pair_end : middle;
  auto const own_begin   = in_left ? pair_begin : middle;
  auto const value       = input[position];

  // Find the number of elements in the other sorted run preceding this element in the stable
  // total order. Every input element therefore computes one unique output position in parallel.
  auto lower = other_begin;
  auto upper = other_end;
  while (lower < upper) {
    auto const probe = lower + (upper - lower) / 2;
    if (stable_less(input[probe], value, comparator)) {
      lower = probe + 1;
    } else {
      upper = probe;
    }
  }
  auto const output_position = pair_begin + (position - own_begin) + (lower - other_begin);
  output[output_position]    = value;
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
 * Run boundaries are marked in parallel, assigned IDs by a scan, and classified without scanning a
 * run serially. Small unresolved runs are completed by bounded insertion sort; large runs use block
 * sorting networks followed by parallel merge-path-style ranking. CUB requires the compacted
 * segment count as a host argument, so only passes that feed a subsequent radix pass transfer that
 * scalar.
 */
template <int bytes_per_pass, bool known_prefix, typename Comparator>
void sorted_order(column_view const& input,
                  mutable_column_view& output,
                  bool ascending,
                  null_order null_precedence,
                  Comparator comparator,
                  segmented_string_sort_config const& tuning,
                  cuda::stream_ref stream)
{
  using key_layout        = radix_key_layout<bytes_per_pass>;
  auto const size         = input.size();
  auto const temp_mr      = cudf::get_current_device_resource_ref();
  auto const strings_view = strings_column_view{input};
  auto const offsets      = strings_view.offsets();
  auto const chars        = strings_view.chars_begin(stream);
  auto strings            = column_device_view::create(input, stream);
  auto const exec         = rmm::exec_policy_nosync(stream, temp_mr);
  auto const valid_size   = key_layout::uses_full_width ? size - input.null_count() : size;
  auto const null_size    = size - valid_size;
  auto indices_a          = rmm::device_uvector<size_type>(valid_size, stream, temp_mr);
  auto indices_b          = rmm::device_uvector<size_type>(valid_size, stream, temp_mr);
  auto null_indices =
    rmm::device_uvector<size_type>(key_layout::uses_full_width ? null_size : 0, stream, temp_mr);
  auto const rows = cuda::counting_iterator<size_type>{0};
  if constexpr (key_layout::uses_full_width) {
    if (null_size == 0) {
      thrust::sequence(exec, indices_a.begin(), indices_a.end(), 0);
    } else {
      thrust::copy_if(exec, rows, rows + size, indices_a.begin(), valid_row_predicate{*strings});
      thrust::copy_if(exec, rows, rows + size, null_indices.begin(), null_row_predicate{*strings});
    }
  } else {
    thrust::sequence(exec, indices_a.begin(), indices_a.end(), 0);
  }

  auto const maximum_radix_passes = static_cast<size_type>(tuning.lexic_precision);
  auto synchronization_points     = size_type{0};
  auto trace_readbacks            = size_type{0};
  auto const radix_run_min        = static_cast<size_type>(tuning.radix_run_min);
  if (tuning.trace) {
    std::fprintf(stderr,
                 "segmented-string-sort bytes=%d precision=%d radix-run-min=%d "
                 "exact-duplicates=%d valid=%d nulls=%d\n",
                 bytes_per_pass,
                 maximum_radix_passes,
                 tuning.radix_run_min,
                 tuning.eliminate_exact_duplicates,
                 valid_size,
                 null_size);
  }

  auto keys_in  = rmm::device_uvector<std::uint64_t>(valid_size, stream, temp_mr);
  auto keys_out = rmm::device_uvector<std::uint64_t>(valid_size, stream, temp_mr);

  auto begins_a = rmm::device_uvector<size_type>(valid_size, stream, temp_mr);
  auto ends_a   = rmm::device_uvector<size_type>(valid_size, stream, temp_mr);
  auto begins_b = rmm::device_uvector<size_type>(valid_size, stream, temp_mr);
  auto ends_b   = rmm::device_uvector<size_type>(valid_size, stream, temp_mr);

  auto active_a   = rmm::device_uvector<std::uint8_t>(valid_size, stream, temp_mr);
  auto active_b   = rmm::device_uvector<std::uint8_t>(valid_size, stream, temp_mr);
  auto run_begins = rmm::device_uvector<size_type>(valid_size, stream, temp_mr);
  auto run_ends   = rmm::device_uvector<size_type>(valid_size, stream, temp_mr);

  // Every recorded final segment contains at least two rows, so half the row count is a tight
  // upper bound.
  auto const maximum_finish_items = (valid_size + 1) / 2;
  auto final_begins       = rmm::device_uvector<size_type>(maximum_finish_items, stream, temp_mr);
  auto final_ends         = rmm::device_uvector<size_type>(maximum_finish_items, stream, temp_mr);
  auto final_prefix_bytes = rmm::device_uvector<size_type>(
    known_prefix ? maximum_finish_items : size_type{0}, stream, temp_mr);
  auto counts = cudf::detail::device_scalar<finish_counts>(finish_counts{}, stream, temp_mr);
  auto refinement =
    cudf::detail::device_scalar<refinement_counts>(refinement_counts{}, stream, temp_mr);

  thrust::fill(exec, begins_a.begin(), begins_a.end(), valid_size);
  thrust::fill(exec, ends_a.begin(), ends_a.end(), valid_size);
  thrust::fill_n(exec, begins_a.begin(), 1, size_type{0});
  thrust::fill_n(exec, ends_a.begin(), 1, valid_size);
  thrust::fill(exec, active_a.begin(), active_a.end(), std::uint8_t{1});

  auto* current_indices        = indices_a.data();
  auto* other_indices          = indices_b.data();
  auto* current_begins         = begins_a.data();
  auto* current_ends           = ends_a.data();
  auto* next_begins            = begins_b.data();
  auto* next_ends              = ends_b.data();
  auto* active                 = active_a.data();
  auto* next_active            = active_b.data();
  auto const config            = cudf::detail::grid_1d{valid_size, 256};
  auto const first_pass_config = cudf::detail::grid_1d{
    valid_size, shuffle_block_size, strings_per_shuffle_cta / shuffle_block_size};
  auto cub_temp_storage = rmm::device_buffer{};
  size_type num_segments{1};
  // This counts algorithm-visible workspace allocations rather than upstream pool growth. It makes
  // stage-to-stage buffer lifetime changes observable without replacing the caller's memory
  // resource.
  auto allocation_count = size_type{24 + (key_layout::uses_full_width ? 1 : 0) +
                                    (known_prefix ? 1 : 0) + (null_indices.size() > 0 ? 1 : 0)};

  for (size_type pass = 0; pass < maximum_radix_passes && num_segments > 0; ++pass) {
    auto active_rows = size_type{-1};
    if (tuning.trace) {
      active_rows =
        static_cast<size_type>(thrust::count(exec, active, active + valid_size, std::uint8_t{1}));
      ++trace_readbacks;
    }
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
            active,
            keys_in.data(),
            valid_size,
            pass * bytes_per_pass);
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
                                                      pass * bytes_per_pass,
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
    thrust::fill(exec, next_active, next_active + valid_size, std::uint8_t{0});
    if (num_segments > 0) {
      mark_active_segments<<<num_segments, 256, 0, stream.get()>>>(
        next_begins, next_ends, num_segments, next_active);
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
    std::swap(active, next_active);
  }

  // Transfer the finish launch metadata together after refinement.
  auto const finish = counts.value(stream);
  ++synchronization_points;
  auto total_chunks = size_type{0};
  if (finish.segments > 0) {
    auto chunk_counts       = rmm::device_uvector<size_type>(finish.segments, stream, temp_mr);
    auto chunk_offsets      = rmm::device_uvector<size_type>(finish.segments, stream, temp_mr);
    auto const chunk_config = cudf::detail::grid_1d{finish.segments, 256};
    compute_finish_chunk_counts<<<chunk_config.num_blocks,
                                  chunk_config.num_threads_per_block,
                                  0,
                                  stream.get()>>>(
      final_begins.data(), final_ends.data(), finish.segments, chunk_counts.data());
    thrust::exclusive_scan(
      exec, chunk_counts.begin(), chunk_counts.end(), chunk_offsets.begin(), size_type{0});
    auto device_total_chunks =
      cudf::detail::device_scalar<size_type>(size_type{0}, stream, temp_mr);
    compute_total_finish_chunks<<<1, 1, 0, stream.get()>>>(
      chunk_offsets.data(), chunk_counts.data(), finish.segments, device_total_chunks.data());
    total_chunks = device_total_chunks.value(stream);
    ++synchronization_points;

    auto chunk_begins       = rmm::device_uvector<size_type>(total_chunks, stream, temp_mr);
    auto chunk_sizes        = rmm::device_uvector<size_type>(total_chunks, stream, temp_mr);
    auto chunk_run_offsets  = rmm::device_uvector<size_type>(total_chunks, stream, temp_mr);
    auto chunk_run_counts   = rmm::device_uvector<size_type>(total_chunks, stream, temp_mr);
    auto chunk_prefix_bytes = rmm::device_uvector<size_type>(total_chunks, stream, temp_mr);
    allocation_count += 8;
    expand_finish_chunks<<<chunk_config.num_blocks,
                           chunk_config.num_threads_per_block,
                           0,
                           stream.get()>>>(final_begins.data(),
                                           final_ends.data(),
                                           final_prefix_bytes.data(),
                                           chunk_offsets.data(),
                                           finish.segments,
                                           chunk_begins.data(),
                                           chunk_sizes.data(),
                                           chunk_run_offsets.data(),
                                           chunk_run_counts.data(),
                                           chunk_prefix_bytes.data());
    bitonic_sort_finish_chunks<<<total_chunks, comparison_chunk_size, 0, stream.get()>>>(
      current_indices,
      other_indices,
      chunk_begins.data(),
      chunk_sizes.data(),
      chunk_prefix_bytes.data(),
      comparator);
    merge_all_sibling_chunks<<<total_chunks, comparison_chunk_size, 0, stream.get()>>>(
      current_indices,
      other_indices,
      chunk_begins.data(),
      chunk_sizes.data(),
      chunk_run_offsets.data(),
      chunk_run_counts.data(),
      chunk_prefix_bytes.data(),
      comparator);
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

  if constexpr (key_layout::uses_full_width) {
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
  } else {
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(
      output.begin<size_type>(), current_indices, sizeof(size_type) * size, stream));
  }
}

}  // namespace segmented_string_sort
}  // namespace cudf::detail
