/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "string_sort_config.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_segmented_radix_sort.cuh>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <utility>

namespace cudf::detail {
namespace segmented_string_sort {

constexpr size_type block_sort_size = 256;
constexpr size_type invalid_index   = -1;

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
  size_type block_tasks{};
  size_type maximum_segment_size{};
};

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

CUDF_KERNEL void scatter_run_endpoints(std::uint8_t const* active,
                                       size_type const* run_starts_at,
                                       std::uint8_t const* run_ends_at,
                                       size_type const* inclusive_run_ids,
                                       size_type size,
                                       size_type* run_begins,
                                       size_type* run_ends)
{
  auto const position = cudf::detail::grid_1d::global_thread_id();
  if (position >= size || active[position] == 0) { return; }
  auto const run = inclusive_run_ids[position] - 1;
  if (run_starts_at[position] != 0) { run_begins[run] = position; }
  if (run_ends_at[position] != 0) { run_ends[run] = position + 1; }
}

template <typename IndexIterator>
CUDF_KERNEL void make_valid_byte_counts(column_device_view strings,
                                        IndexIterator indices,
                                        std::uint8_t const* active,
                                        size_type size,
                                        size_type byte_offset,
                                        std::uint8_t* valid_byte_counts)
{
  auto const position = cudf::detail::grid_1d::global_thread_id();
  if (position >= size || active[position] == 0) { return; }
  auto const value_size       = strings.element<string_view>(indices[position]).size_bytes();
  auto const remaining        = value_size > byte_offset ? value_size - byte_offset : size_type{0};
  auto const bytes            = remaining < static_cast<size_type>(sizeof(std::uint64_t))
                                  ? remaining
                                  : static_cast<size_type>(sizeof(std::uint64_t));
  valid_byte_counts[position] = static_cast<std::uint8_t>(bytes);
}

CUDF_KERNEL void reduce_minimum_run_bytes(std::uint8_t const* active,
                                          size_type const* inclusive_run_ids,
                                          std::uint8_t const* valid_byte_counts,
                                          size_type size,
                                          size_type* minimum_run_bytes)
{
  auto const position = cudf::detail::grid_1d::global_thread_id();
  if (position >= size || active[position] == 0) { return; }
  atomicMin(minimum_run_bytes + inclusive_run_ids[position] - 1,
            static_cast<size_type>(valid_byte_counts[position]));
}

template <int bytes_per_pass>
__device__ size_type proven_prefix_bytes(size_type run,
                                         size_type byte_offset,
                                         size_type const* minimum_run_bytes)
{
  using key_layout = radix_key_layout<bytes_per_pass>;
  if constexpr (key_layout::stores_metadata) {
    return byte_offset + bytes_per_pass;
  } else {
    return minimum_run_bytes[run] == bytes_per_pass ? byte_offset + bytes_per_pass : byte_offset;
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
                                        size_type const* minimum_run_bytes,
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
    proven_prefix_bytes<bytes_per_pass>(run, byte_offset, minimum_run_bytes);
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
                                     size_type const* minimum_run_bytes,
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
    proven_prefix_bytes<bytes_per_pass>(run, byte_offset, minimum_run_bytes);
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
                               size_type const* minimum_run_bytes,
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
                           : minimum_run_bytes[run];
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
      proven_prefix_bytes<bytes_per_pass>(run, byte_offset, minimum_run_bytes);
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
  using key_layout      = radix_key_layout<bytes_per_pass>;
  auto const size       = input.size();
  auto const temp_mr    = cudf::get_current_device_resource_ref();
  auto strings          = column_device_view::create(input, stream);
  auto const exec       = rmm::exec_policy_nosync(stream, temp_mr);
  auto const valid_size = key_layout::uses_full_width ? size - input.null_count() : size;
  auto const null_size  = size - valid_size;
  auto indices_a        = rmm::device_uvector<size_type>(valid_size, stream, temp_mr);
  auto indices_b        = rmm::device_uvector<size_type>(valid_size, stream, temp_mr);
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

  auto max_length             = size_type{-1};
  auto maximum_radix_passes   = static_cast<size_type>(tuning.max_radix_passes);
  auto synchronization_points = size_type{0};
  auto trace_readbacks        = size_type{0};
  if (tuning.max_radix_passes == 0 || tuning.radix_percent != 100) {
    max_length = thrust::transform_reduce(exec,
                                          indices_a.begin(),
                                          indices_a.end(),
                                          string_length_fn{*strings},
                                          size_type{0},
                                          cuda::maximum<size_type>{});
    ++synchronization_points;
    auto const target_bytes = std::max<size_type>(
      1,
      static_cast<size_type>((static_cast<std::int64_t>(max_length) * tuning.radix_percent + 99) /
                             100));
    maximum_radix_passes =
      std::max<size_type>(1, (target_bytes + bytes_per_pass - 1) / bytes_per_pass);
    if (tuning.max_radix_passes != 0) {
      maximum_radix_passes =
        std::min(maximum_radix_passes, static_cast<size_type>(tuning.max_radix_passes));
    }
  }
  auto const comparison_threshold = static_cast<size_type>(tuning.finish_threshold);
  if (tuning.trace) {
    std::fprintf(stderr,
                 "segmented-string-sort bytes=%d percent=%d passes=%d known-prefix=%d "
                 "finish-threshold=%d rle-policy=%d valid=%d nulls=%d max-length=%d\n",
                 bytes_per_pass,
                 tuning.radix_percent,
                 maximum_radix_passes,
                 tuning.known_prefix,
                 tuning.finish_threshold,
                 static_cast<int>(tuning.rle_policy),
                 valid_size,
                 null_size,
                 max_length);
  }

  auto keys_in  = rmm::device_uvector<std::uint64_t>(valid_size, stream, temp_mr);
  auto keys_out = rmm::device_uvector<std::uint64_t>(valid_size, stream, temp_mr);

  auto begins_a = rmm::device_uvector<size_type>(valid_size, stream, temp_mr);
  auto ends_a   = rmm::device_uvector<size_type>(valid_size, stream, temp_mr);
  auto begins_b = rmm::device_uvector<size_type>(valid_size, stream, temp_mr);
  auto ends_b   = rmm::device_uvector<size_type>(valid_size, stream, temp_mr);

  auto active_a          = rmm::device_uvector<std::uint8_t>(valid_size, stream, temp_mr);
  auto active_b          = rmm::device_uvector<std::uint8_t>(valid_size, stream, temp_mr);
  auto segment_starts    = rmm::device_uvector<std::uint8_t>(valid_size, stream, temp_mr);
  auto segment_ends_at   = rmm::device_uvector<std::uint8_t>(valid_size, stream, temp_mr);
  auto run_starts_at     = rmm::device_uvector<size_type>(valid_size, stream, temp_mr);
  auto run_ends_at       = rmm::device_uvector<std::uint8_t>(valid_size, stream, temp_mr);
  auto run_ids           = rmm::device_uvector<size_type>(valid_size, stream, temp_mr);
  auto run_begins        = rmm::device_uvector<size_type>(valid_size, stream, temp_mr);
  auto run_ends          = rmm::device_uvector<size_type>(valid_size, stream, temp_mr);
  auto valid_byte_counts = rmm::device_uvector<std::uint8_t>(
    key_layout::uses_full_width ? valid_size : size_type{0}, stream, temp_mr);
  auto minimum_run_bytes = rmm::device_uvector<size_type>(
    key_layout::uses_full_width ? valid_size : size_type{0}, stream, temp_mr);

  // Every recorded final segment and block task contains at least two rows, so half the row count
  // is a tight upper bound for each collection.
  auto const maximum_finish_items = (valid_size + 1) / 2;
  auto final_begins       = rmm::device_uvector<size_type>(maximum_finish_items, stream, temp_mr);
  auto final_ends         = rmm::device_uvector<size_type>(maximum_finish_items, stream, temp_mr);
  auto final_prefix_bytes = rmm::device_uvector<size_type>(
    known_prefix ? maximum_finish_items : size_type{0}, stream, temp_mr);
  auto task_begins = rmm::device_uvector<size_type>(maximum_finish_items, stream, temp_mr);
  auto task_ends   = rmm::device_uvector<size_type>(maximum_finish_items, stream, temp_mr);
  auto counts      = cudf::detail::device_scalar<finish_counts>(finish_counts{}, stream, temp_mr);
  auto next_segment_count = cudf::detail::device_scalar<size_type>(size_type{0}, stream, temp_mr);

  thrust::fill(exec, begins_a.begin(), begins_a.end(), valid_size);
  thrust::fill(exec, ends_a.begin(), ends_a.end(), valid_size);
  thrust::fill_n(exec, begins_a.begin(), 1, size_type{0});
  thrust::fill_n(exec, ends_a.begin(), 1, valid_size);
  thrust::fill(exec, active_a.begin(), active_a.end(), std::uint8_t{1});

  auto* current_indices = indices_a.data();
  auto* other_indices   = indices_b.data();
  auto* current_begins  = begins_a.data();
  auto* current_ends    = ends_a.data();
  auto* next_begins     = begins_b.data();
  auto* next_ends       = ends_b.data();
  auto* active          = active_a.data();
  auto* next_active     = active_b.data();
  auto const config     = cudf::detail::grid_1d{valid_size, 256};
  auto cub_temp_storage = rmm::device_buffer{};
  size_type num_segments{1};
  // This counts algorithm-visible workspace allocations rather than upstream pool growth. It makes
  // stage-to-stage buffer lifetime changes observable without replacing the caller's memory
  // resource.
  auto allocation_count = size_type{24 + (key_layout::uses_full_width ? 2 : 0) +
                                    (known_prefix ? 1 : 0) + (null_indices.size() > 0 ? 1 : 0)};

  for (size_type pass = 0; pass < maximum_radix_passes && num_segments > 0; ++pass) {
    auto active_rows = size_type{-1};
    if (tuning.trace) {
      active_rows =
        static_cast<size_type>(thrust::count(exec, active, active + valid_size, std::uint8_t{1}));
      ++trace_readbacks;
    }
    thrust::fill(exec, segment_starts.begin(), segment_starts.end(), std::uint8_t{0});
    thrust::fill(exec, segment_ends_at.begin(), segment_ends_at.end(), std::uint8_t{0});
    thrust::fill(exec, run_starts_at.begin(), run_starts_at.end(), size_type{0});
    thrust::fill(exec, run_ends_at.begin(), run_ends_at.end(), std::uint8_t{0});
    thrust::fill(exec, run_begins.begin(), run_begins.end(), invalid_index);
    thrust::fill(exec, run_ends.begin(), run_ends.end(), invalid_index);
    thrust::fill(exec, next_begins, next_begins + valid_size, valid_size);
    thrust::fill(exec, next_ends, next_ends + valid_size, valid_size);

    make_keys<bytes_per_pass>
      <<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(*strings,
                                                                             current_indices,
                                                                             active,
                                                                             keys_in.data(),
                                                                             valid_size,
                                                                             pass * bytes_per_pass,
                                                                             null_precedence);
    CUDF_CUDA_TRY(cudaGetLastError());
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(
      other_indices, current_indices, sizeof(size_type) * valid_size, stream));
    auto const cub_storage_was_empty = cub_temp_storage.size() == 0;
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
    if (cub_storage_was_empty && cub_temp_storage.size() > 0) { ++allocation_count; }
    std::swap(current_indices, other_indices);

    auto const segment_config = cudf::detail::grid_1d{num_segments, 256};
    mark_segment_endpoints<<<segment_config.num_blocks,
                             segment_config.num_threads_per_block,
                             0,
                             stream.get()>>>(
      current_begins, current_ends, num_segments, segment_starts.data(), segment_ends_at.data());
    mark_run_boundaries<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
      active,
      segment_starts.data(),
      segment_ends_at.data(),
      keys_out.data(),
      valid_size,
      run_starts_at.data(),
      run_ends_at.data());
    CUDF_CUDA_TRY(cudaGetLastError());

    thrust::inclusive_scan(exec, run_starts_at.begin(), run_starts_at.end(), run_ids.begin());
    scatter_run_endpoints<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
      active,
      run_starts_at.data(),
      run_ends_at.data(),
      run_ids.data(),
      valid_size,
      run_begins.data(),
      run_ends.data());
    if constexpr (key_layout::uses_full_width) {
      thrust::fill(
        exec, minimum_run_bytes.begin(), minimum_run_bytes.end(), size_type{bytes_per_pass});
      make_valid_byte_counts<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
        *strings,
        current_indices,
        active,
        valid_size,
        pass * bytes_per_pass,
        valid_byte_counts.data());
      reduce_minimum_run_bytes<<<config.num_blocks,
                                 config.num_threads_per_block,
                                 0,
                                 stream.get()>>>(
        active, run_ids.data(), valid_byte_counts.data(), valid_size, minimum_run_bytes.data());
    }
    auto run_count = size_type{-1};
    if (tuning.trace) {
      run_count = static_cast<size_type>(
        thrust::count(exec, run_ends_at.begin(), run_ends_at.end(), std::uint8_t{1}));
      ++trace_readbacks;
    }
    auto const last_pass   = pass + 1 == maximum_radix_passes;
    auto detect_duplicates = last_pass && tuning.rle_policy == segmented_rle_policy::ALWAYS;
    if (last_pass && tuning.rle_policy == segmented_rle_policy::ADAPTIVE) {
      auto metrics = cudf::detail::device_scalar<rle_metrics>(rle_metrics{}, stream, temp_mr);
      ++allocation_count;
      auto const sampling_stride = std::max<size_type>(1, (valid_size + 4095) / 4096);
      collect_rle_metrics<bytes_per_pass>
        <<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
          *strings,
          current_indices,
          active,
          run_ids.data(),
          run_begins.data(),
          run_ends.data(),
          valid_size,
          tuning.rle_min_run_length,
          sampling_stride,
          pass * bytes_per_pass,
          key_layout::uses_full_width ? minimum_run_bytes.data() : nullptr,
          metrics.data());
      auto const observed = metrics.value(stream);
      ++synchronization_points;
      auto const coverage = valid_size == 0 ? 0ULL : observed.covered_rows * 100 / valid_size;
      auto const equality =
        observed.sampled_pairs == 0 ? 0ULL : observed.equal_pairs * 100 / observed.sampled_pairs;
      detect_duplicates = coverage >= static_cast<unsigned>(tuning.rle_min_coverage_percent) &&
                          equality >= static_cast<unsigned>(tuning.rle_min_equal_percent);
      if (tuning.trace) {
        std::fprintf(stderr,
                     "segmented-string-sort adaptive-rle coverage=%llu equality=%llu samples=%llu "
                     "enabled=%d\n",
                     coverage,
                     equality,
                     observed.sampled_pairs,
                     detect_duplicates);
      }
    }
    thrust::fill(exec, run_starts_at.begin(), run_starts_at.end(), size_type{0});
    if (detect_duplicates) {
      mark_nonduplicate_runs<bytes_per_pass>
        <<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
          *strings,
          current_indices,
          active,
          run_ids.data(),
          run_begins.data(),
          valid_size,
          pass * bytes_per_pass,
          key_layout::uses_full_width ? minimum_run_bytes.data() : nullptr,
          run_starts_at.data());
    }
    // The positional start flags are no longer needed. Reuse their storage for mismatch markers,
    // per-run next flags, and later compacted next-segment IDs.
    classify_runs<bytes_per_pass>
      <<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
        *strings,
        current_indices,
        keys_out.data(),
        run_begins.data(),
        run_ends.data(),
        key_layout::uses_full_width ? minimum_run_bytes.data() : nullptr,
        valid_size,
        pass * bytes_per_pass,
        last_pass,
        detect_duplicates,
        detect_duplicates ? run_starts_at.data() : nullptr,
        comparison_threshold,
        run_starts_at.data(),
        final_begins.data(),
        final_ends.data(),
        known_prefix ? final_prefix_bytes.data() : nullptr,
        task_begins.data(),
        task_ends.data(),
        counts.data());
    update_active_runs<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
      active, run_ids.data(), run_starts_at.data(), valid_size, next_active);
    thrust::inclusive_scan(exec, run_starts_at.begin(), run_starts_at.end(), run_ids.begin());
    compact_next_segments<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
      run_begins.data(),
      run_ends.data(),
      run_starts_at.data(),
      run_ids.data(),
      valid_size,
      next_begins,
      next_ends);
    CUDF_CUDA_TRY(cudaGetLastError());

    auto const needs_next_segment_count = pass + 1 < maximum_radix_passes;
    if (needs_next_segment_count || tuning.trace) {
      CUDF_CUDA_TRY(cudf::detail::memcpy_async(
        next_segment_count.data(), run_ids.data() + valid_size - 1, sizeof(size_type), stream));
      auto const next_segments = next_segment_count.value(stream);
      if (needs_next_segment_count) {
        num_segments = next_segments;
        ++synchronization_points;
      } else {
        ++trace_readbacks;
      }
      if (tuning.trace) {
        std::fprintf(stderr,
                     "segmented-string-sort pass=%d active-rows=%d runs=%d next-runs=%d\n",
                     pass,
                     active_rows,
                     run_count,
                     next_segments);
      }
    }

    std::swap(current_begins, next_begins);
    std::swap(current_ends, next_ends);
    std::swap(active, next_active);
  }

  // Transfer the finish launch metadata together after refinement.
  auto const finish = counts.value(stream);
  ++synchronization_points;
  auto known_prefix_by_row = rmm::device_uvector<size_type>(
    known_prefix && finish.segments > 0 ? size : size_type{0}, stream, temp_mr);
  allocation_count += known_prefix_by_row.size() > 0 ? 1 : 0;
  auto merge_levels = size_type{0};
  if (known_prefix_by_row.size() != 0) {
    thrust::fill(exec, known_prefix_by_row.begin(), known_prefix_by_row.end(), size_type{0});
    if constexpr (known_prefix) {
      comparator.transform.known_prefix_bytes = known_prefix_by_row.data();
    }
  }
  if (finish.segments > 0) {
    auto final_begin_for_position = rmm::device_uvector<size_type>(valid_size, stream, temp_mr);
    auto final_end_for_position   = rmm::device_uvector<size_type>(valid_size, stream, temp_mr);
    allocation_count += 2;
    thrust::fill(
      exec, final_begin_for_position.begin(), final_begin_for_position.end(), invalid_index);
    thrust::fill(exec, final_end_for_position.begin(), final_end_for_position.end(), invalid_index);
    map_final_segments<<<finish.segments, 256, 0, stream.get()>>>(
      final_begins.data(),
      final_ends.data(),
      known_prefix ? final_prefix_bytes.data() : nullptr,
      finish.segments,
      current_indices,
      final_begin_for_position.data(),
      final_end_for_position.data(),
      known_prefix_by_row.data());

    auto const finish_config = cudf::detail::grid_1d{finish.segments, 128};
    finish_small_segments<<<finish_config.num_blocks,
                            finish_config.num_threads_per_block,
                            0,
                            stream.get()>>>(current_indices,
                                            final_begins.data(),
                                            final_ends.data(),
                                            finish.segments,
                                            comparison_threshold,
                                            comparator);
    if (finish.block_tasks > 0) {
      block_sort_tasks<<<finish.block_tasks, block_sort_size, 0, stream.get()>>>(
        current_indices, task_begins.data(), task_ends.data(), finish.block_tasks, comparator);
      for (std::int64_t width = block_sort_size; width < finish.maximum_segment_size; width *= 2) {
        ++merge_levels;
        merge_sorted_blocks<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
          current_indices,
          other_indices,
          final_begin_for_position.data(),
          final_end_for_position.data(),
          valid_size,
          width,
          comparison_threshold,
          comparator);
        std::swap(current_indices, other_indices);
      }
    }
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  if (tuning.trace) {
    std::fprintf(stderr,
                 "segmented-string-sort finish-segments=%d finish-rows=%d block-tasks=%d "
                 "merge-levels=%d allocations=%d sync-points=%d trace-readbacks=%d\n",
                 finish.segments,
                 finish.rows,
                 finish.block_tasks,
                 merge_levels,
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
