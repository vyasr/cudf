/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "sort.hpp"

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
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

#include <cstdint>
#include <utility>

namespace cudf::detail {
namespace segmented_string_sort {

// Six bytes leave room in a uint64_t key for a null category and the number of valid bytes.
constexpr size_type bytes_per_pass       = 6;
constexpr size_type maximum_radix_passes = 4;
constexpr size_type comparison_threshold = 32;
constexpr size_type block_sort_size      = 256;
constexpr size_type invalid_index        = -1;

struct finish_counts {
  size_type segments{};
  size_type block_tasks{};
};

template <typename IndexIterator>
CUDF_KERNEL void make_keys(column_device_view strings,
                           IndexIterator indices,
                           std::uint8_t const* active,
                           std::uint64_t* keys,
                           size_type size,
                           size_type byte_offset,
                           null_order null_precedence)
{
  auto const position = cudf::detail::grid_1d::global_thread_id();
  if (position >= size || active[position] == 0) { return; }

  auto const row = indices[position];
  // Category zero sorts before category one, while category two sorts after it. Descending radix
  // sort reverses all three categories, matching libcudf's existing null-order semantics.
  if (strings.is_null(row)) {
    auto const category =
      null_precedence == null_order::BEFORE ? std::uint64_t{0} : std::uint64_t{2};
    keys[position] = category << 62;
    return;
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
  keys[position] = (std::uint64_t{1} << 62) | (bytes << 3) | length;
}

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
CUDF_KERNEL void mark_nonduplicate_runs(column_device_view strings,
                                        IndexIterator indices,
                                        std::uint8_t const* active,
                                        size_type const* inclusive_run_ids,
                                        size_type const* run_begins,
                                        size_type size,
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
  auto const equal =
    row_null == first_null &&
    (row_null || strings.element<string_view>(row) == strings.element<string_view>(first_row));
  if (!equal) { atomicExch(run_has_mismatch + run, size_type{1}); }
}

template <bool detect_duplicates, typename IndexIterator>
CUDF_KERNEL void classify_runs(column_device_view strings,
                               IndexIterator indices,
                               std::uint64_t const* sorted_keys,
                               size_type const* run_begins,
                               size_type const* run_ends,
                               size_type num_slots,
                               bool last_pass,
                               size_type const* run_has_mismatch,
                               size_type* next_run_flags,
                               std::uint8_t* run_classes,
                               size_type* final_begins,
                               size_type* final_ends,
                               size_type* task_begins,
                               size_type* task_ends,
                               finish_counts* counts)
{
  auto const run = cudf::detail::grid_1d::global_thread_id();
  if (run >= num_slots) { return; }
  auto const begin = run_begins[run];
  auto const end   = run_ends[run];
  if (begin == invalid_index || end - begin <= 1) { return; }

  auto const row       = indices[begin];
  auto const is_null   = strings.is_null(row);
  auto const key_bytes = static_cast<size_type>(sorted_keys[begin] & 0x7);
  // A short equal chunk means the entire run consists of exact duplicates. Stable radix sorting
  // has already put it in its final order, including null runs.
  if (is_null || key_bytes != bytes_per_pass) { return; }

  // The final radix chunk cannot distinguish long exact duplicates from strings that differ
  // later. Exact duplicate runs are already stable and need no comparison-sort finish.
  if constexpr (detect_duplicates) {
    if (last_pass && run_has_mismatch[run] == 0) { return; }
  }

  if (!last_pass && end - begin > comparison_threshold) {
    next_run_flags[run] = 1;
    run_classes[run]    = 1;
    return;
  }

  run_classes[run]         = 2;
  auto const final_slot    = atomicAdd(&counts->segments, size_type{1});
  final_begins[final_slot] = begin;
  final_ends[final_slot]   = end;
  if (end - begin > comparison_threshold) {
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
                                    std::uint8_t const* run_classes,
                                    size_type size,
                                    std::uint8_t* next_active)
{
  auto const position = cudf::detail::grid_1d::global_thread_id();
  if (position >= size) { return; }
  next_active[position] =
    active[position] != 0 && run_classes[inclusive_run_ids[position] - 1] == 1 ? 1 : 0;
}

CUDF_KERNEL void map_final_segments(size_type const* final_begins,
                                    size_type const* final_ends,
                                    size_type num_segments,
                                    size_type* final_begin_for_position,
                                    size_type* final_end_for_position)
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
                                       Comparator comparator)
{
  auto const segment = cudf::detail::grid_1d::global_thread_id();
  if (segment >= num_segments) { return; }
  auto const begin = final_begins[segment];
  auto const end   = final_ends[segment];
  if (end - begin > comparison_threshold) { return; }

  // This serial path is strictly bounded to 32 elements. Tie-breaking by original row index makes
  // the result stable while remaining valid for the unstable API.
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
  auto temp_storage =
    rmm::device_buffer(temp_storage_bytes, stream, cudf::get_current_device_resource_ref());
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
template <sort_method method, bool detect_duplicates, typename Comparator>
void sorted_order(column_view const& input,
                  mutable_column_view& output,
                  bool ascending,
                  null_order null_precedence,
                  Comparator comparator,
                  cuda::stream_ref stream)
{
  auto const size = input.size();
  if (size == 0) { return; }

  auto const temp_mr = cudf::get_current_device_resource_ref();
  auto strings       = column_device_view::create(input, stream);
  auto indices_a     = rmm::device_uvector<size_type>(size, stream, temp_mr);
  auto indices_b     = rmm::device_uvector<size_type>(size, stream, temp_mr);
  auto keys_in       = rmm::device_uvector<std::uint64_t>(size, stream, temp_mr);
  auto keys_out      = rmm::device_uvector<std::uint64_t>(size, stream, temp_mr);

  auto begins_a = rmm::device_uvector<size_type>(size, stream, temp_mr);
  auto ends_a   = rmm::device_uvector<size_type>(size, stream, temp_mr);
  auto begins_b = rmm::device_uvector<size_type>(size, stream, temp_mr);
  auto ends_b   = rmm::device_uvector<size_type>(size, stream, temp_mr);

  auto active_a        = rmm::device_uvector<std::uint8_t>(size, stream, temp_mr);
  auto active_b        = rmm::device_uvector<std::uint8_t>(size, stream, temp_mr);
  auto segment_starts  = rmm::device_uvector<std::uint8_t>(size, stream, temp_mr);
  auto segment_ends_at = rmm::device_uvector<std::uint8_t>(size, stream, temp_mr);
  auto run_starts_at   = rmm::device_uvector<size_type>(size, stream, temp_mr);
  auto run_ends_at     = rmm::device_uvector<std::uint8_t>(size, stream, temp_mr);
  auto run_ids         = rmm::device_uvector<size_type>(size, stream, temp_mr);
  auto run_begins      = rmm::device_uvector<size_type>(size, stream, temp_mr);
  auto run_ends        = rmm::device_uvector<size_type>(size, stream, temp_mr);
  auto run_classes     = rmm::device_uvector<std::uint8_t>(size, stream, temp_mr);
  auto run_has_mismatch =
    rmm::device_uvector<size_type>(detect_duplicates ? size : 0, stream, temp_mr);

  auto final_begins = rmm::device_uvector<size_type>(size, stream, temp_mr);
  auto final_ends   = rmm::device_uvector<size_type>(size, stream, temp_mr);
  auto task_begins  = rmm::device_uvector<size_type>(size, stream, temp_mr);
  auto task_ends    = rmm::device_uvector<size_type>(size, stream, temp_mr);
  auto counts       = cudf::detail::device_scalar<finish_counts>(finish_counts{}, stream, temp_mr);
  auto next_segment_count = cudf::detail::device_scalar<size_type>(size_type{0}, stream, temp_mr);

  auto const exec = rmm::exec_policy_nosync(stream, temp_mr);
  thrust::sequence(exec, indices_a.begin(), indices_a.end(), 0);
  thrust::fill(exec, begins_a.begin(), begins_a.end(), size);
  thrust::fill(exec, ends_a.begin(), ends_a.end(), size);
  thrust::fill_n(exec, begins_a.begin(), 1, size_type{0});
  thrust::fill_n(exec, ends_a.begin(), 1, size);
  thrust::fill(exec, active_a.begin(), active_a.end(), std::uint8_t{1});

  auto* current_indices = indices_a.data();
  auto* other_indices   = indices_b.data();
  auto* current_begins  = begins_a.data();
  auto* current_ends    = ends_a.data();
  auto* next_begins     = begins_b.data();
  auto* next_ends       = ends_b.data();
  auto* active          = active_a.data();
  auto* next_active     = active_b.data();
  auto const config     = cudf::detail::grid_1d{size, 256};
  size_type num_segments{1};

  for (size_type pass = 0; pass < maximum_radix_passes && num_segments > 0; ++pass) {
    thrust::fill(exec, segment_starts.begin(), segment_starts.end(), std::uint8_t{0});
    thrust::fill(exec, segment_ends_at.begin(), segment_ends_at.end(), std::uint8_t{0});
    thrust::fill(exec, run_starts_at.begin(), run_starts_at.end(), size_type{0});
    thrust::fill(exec, run_ends_at.begin(), run_ends_at.end(), std::uint8_t{0});
    thrust::fill(exec, run_begins.begin(), run_begins.end(), invalid_index);
    thrust::fill(exec, run_ends.begin(), run_ends.end(), invalid_index);
    thrust::fill(exec, run_classes.begin(), run_classes.end(), std::uint8_t{0});
    thrust::fill(exec, next_begins, next_begins + size, size);
    thrust::fill(exec, next_ends, next_ends + size, size);

    make_keys<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
      *strings,
      current_indices,
      active,
      keys_in.data(),
      size,
      pass * bytes_per_pass,
      null_precedence);
    CUDF_CUDA_TRY(cudaGetLastError());
    CUDF_CUDA_TRY(
      cudf::detail::memcpy_async(other_indices, current_indices, sizeof(size_type) * size, stream));
    segmented_radix_sort(keys_in.data(),
                         keys_out.data(),
                         current_indices,
                         other_indices,
                         size,
                         num_segments,
                         current_begins,
                         current_ends,
                         ascending,
                         stream);
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
      size,
      run_starts_at.data(),
      run_ends_at.data());
    CUDF_CUDA_TRY(cudaGetLastError());

    thrust::inclusive_scan(exec, run_starts_at.begin(), run_starts_at.end(), run_ids.begin());
    scatter_run_endpoints<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
      active,
      run_starts_at.data(),
      run_ends_at.data(),
      run_ids.data(),
      size,
      run_begins.data(),
      run_ends.data());
    auto const last_pass = pass + 1 == maximum_radix_passes;
    if constexpr (detect_duplicates) {
      if (last_pass) {
        thrust::fill(exec, run_has_mismatch.begin(), run_has_mismatch.end(), size_type{0});
        mark_nonduplicate_runs<<<config.num_blocks,
                                 config.num_threads_per_block,
                                 0,
                                 stream.get()>>>(*strings,
                                                 current_indices,
                                                 active,
                                                 run_ids.data(),
                                                 run_begins.data(),
                                                 size,
                                                 run_has_mismatch.data());
      }
    }
    // The positional start flags are no longer needed. Reuse their storage for the per-run next
    // flags and later for compacted next-segment IDs.
    thrust::fill(exec, run_starts_at.begin(), run_starts_at.end(), size_type{0});
    classify_runs<detect_duplicates>
      <<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
        *strings,
        current_indices,
        keys_out.data(),
        run_begins.data(),
        run_ends.data(),
        size,
        last_pass,
        run_has_mismatch.data(),
        run_starts_at.data(),
        run_classes.data(),
        final_begins.data(),
        final_ends.data(),
        task_begins.data(),
        task_ends.data(),
        counts.data());
    update_active_runs<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
      active, run_ids.data(), run_classes.data(), size, next_active);
    thrust::inclusive_scan(exec, run_starts_at.begin(), run_starts_at.end(), run_ids.begin());
    compact_next_segments<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
      run_begins.data(),
      run_ends.data(),
      run_starts_at.data(),
      run_ids.data(),
      size,
      next_begins,
      next_ends);
    CUDF_CUDA_TRY(cudaGetLastError());

    if (pass + 1 < maximum_radix_passes) {
      CUDF_CUDA_TRY(cudf::detail::memcpy_async(
        next_segment_count.data(), run_ids.data() + size - 1, sizeof(size_type), stream));
      num_segments = next_segment_count.value(stream);
    }

    std::swap(current_begins, next_begins);
    std::swap(current_ends, next_ends);
    std::swap(active, next_active);
  }

  // Transfer both finish launch counts together after refinement.
  auto const finish = counts.value(stream);
  if (finish.segments > 0) {
    auto final_begin_for_position = rmm::device_uvector<size_type>(size, stream, temp_mr);
    auto final_end_for_position   = rmm::device_uvector<size_type>(size, stream, temp_mr);
    thrust::fill(
      exec, final_begin_for_position.begin(), final_begin_for_position.end(), invalid_index);
    thrust::fill(exec, final_end_for_position.begin(), final_end_for_position.end(), invalid_index);
    map_final_segments<<<finish.segments, 256, 0, stream.get()>>>(final_begins.data(),
                                                                  final_ends.data(),
                                                                  finish.segments,
                                                                  final_begin_for_position.data(),
                                                                  final_end_for_position.data());

    auto const finish_config = cudf::detail::grid_1d{finish.segments, 128};
    finish_small_segments<<<finish_config.num_blocks,
                            finish_config.num_threads_per_block,
                            0,
                            stream.get()>>>(
      current_indices, final_begins.data(), final_ends.data(), finish.segments, comparator);
    if (finish.block_tasks > 0) {
      block_sort_tasks<<<finish.block_tasks, block_sort_size, 0, stream.get()>>>(
        current_indices, task_begins.data(), task_ends.data(), finish.block_tasks, comparator);
      for (std::int64_t width = block_sort_size; width < size; width *= 2) {
        merge_sorted_blocks<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
          current_indices,
          other_indices,
          final_begin_for_position.data(),
          final_end_for_position.data(),
          size,
          width,
          comparator);
        std::swap(current_indices, other_indices);
      }
    }
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  CUDF_CUDA_TRY(cudf::detail::memcpy_async(
    output.begin<size_type>(), current_indices, sizeof(size_type) * size, stream));
}

}  // namespace segmented_string_sort
}  // namespace cudf::detail
