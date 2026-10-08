/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace cudf::detail {

// Preserve the Thrust dispatch independently of the direct CUB launch variants.
void sort_int_keys(int* begin,
                   int* end,
                   cuda::stream_ref stream,
                   rmm::device_async_resource_ref mr);

// Query and execution retain caller-owned storage and the original 32-bit count.
cudaError_t radix_sort_int_keys(void* storage,
                                std::size_t& storage_bytes,
                                int const* keys_in,
                                int* keys_out,
                                int count,
                                int begin_bit,
                                int end_bit,
                                cudaStream_t stream);

// The execution-environment overload retains CUB-managed storage and 64-bit counts.
cudaError_t radix_sort_int_keys(int const* keys_in,
                                int* keys_out,
                                std::int64_t count,
                                int begin_bit,
                                int end_bit,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr);
}  // namespace cudf::detail
