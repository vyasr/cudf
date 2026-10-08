/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "radix_sort_keys.hpp"

#include <rmm/exec_policy.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cuda/std/execution>
#include <thrust/sort.h>

namespace cudf::detail {

void sort_int_keys(int* begin, int* end, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  thrust::sort(rmm::exec_policy_nosync(stream, mr), begin, end);
}

cudaError_t radix_sort_int_keys(void* storage,
                                std::size_t& storage_bytes,
                                int const* keys_in,
                                int* keys_out,
                                int count,
                                int begin_bit,
                                int end_bit,
                                cudaStream_t stream)
{
  return cub::DeviceRadixSort::SortKeys(
    storage, storage_bytes, keys_in, keys_out, count, begin_bit, end_bit, stream);
}

cudaError_t radix_sort_int_keys(int const* keys_in,
                                int* keys_out,
                                std::int64_t count,
                                int begin_bit,
                                int end_bit,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr)
{
  auto const mr_prop = cuda::std::execution::prop{cuda::mr::get_memory_resource, mr};
  auto const env     = cuda::std::execution::env{cuda::stream_ref{stream.get()}, mr_prop};
  return cub::DeviceRadixSort::SortKeys(keys_in, keys_out, count, begin_bit, end_bit, env);
}

}  // namespace cudf::detail
