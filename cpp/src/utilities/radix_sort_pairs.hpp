/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/memory_resource.hpp>

#include <cuda/std/span>
#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <cstddef>

namespace cudf::detail {

// Keep the existing Thrust entry points distinct while sharing their device compilation owner.
void sort_int_pairs(
  int* begin, int* end, int* values, cuda::stream_ref stream, rmm::device_async_resource_ref mr);
void stable_sort_int_pairs(
  int* begin, int* end, int* values, cuda::stream_ref stream, rmm::device_async_resource_ref mr);
// Span callers retain their existing Thrust dispatch while sharing device compilation ownership.
void stable_sort_int_pairs(cuda::std::span<int>::iterator begin,
                           cuda::std::span<int>::iterator end,
                           cuda::std::span<int>::iterator values,
                           cuda::stream_ref stream,
                           rmm::device_async_resource_ref mr);
void stable_sort_int_pairs_less(
  int* begin, int* end, int* values, cuda::stream_ref stream, rmm::device_async_resource_ref mr);

// Delegating only the launch keeps CUB callers in control of temporary storage and bit ranges.
cudaError_t radix_sort_int_pairs(void* storage,
                                 std::size_t& storage_bytes,
                                 int const* keys_in,
                                 int* keys_out,
                                 int const* values_in,
                                 int* values_out,
                                 int count,
                                 int begin_bit,
                                 int end_bit,
                                 cudaStream_t stream);
}  // namespace cudf::detail
