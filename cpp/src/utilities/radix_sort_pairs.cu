/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "radix_sort_pairs.hpp"

#include <rmm/exec_policy.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cuda/std/functional>
#include <thrust/sort.h>

namespace cudf::detail {

void sort_int_pairs(
  int* begin, int* end, int* values, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  thrust::sort_by_key(rmm::exec_policy_nosync(stream, mr), begin, end, values);
}

void stable_sort_int_pairs(
  int* begin, int* end, int* values, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  thrust::stable_sort_by_key(rmm::exec_policy_nosync(stream, mr), begin, end, values);
}

void stable_sort_int_pairs(cuda::std::span<int>::iterator begin,
                           cuda::std::span<int>::iterator end,
                           cuda::std::span<int>::iterator values,
                           cuda::stream_ref stream,
                           rmm::device_async_resource_ref mr)
{
  thrust::stable_sort_by_key(rmm::exec_policy_nosync(stream, mr), begin, end, values);
}

void stable_sort_int_pairs_less(
  int* begin, int* end, int* values, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  thrust::stable_sort_by_key(
    rmm::exec_policy_nosync(stream, mr), begin, end, values, cuda::std::less<int>{});
}

cudaError_t radix_sort_int_pairs(void* storage,
                                 std::size_t& storage_bytes,
                                 int const* keys_in,
                                 int* keys_out,
                                 int const* values_in,
                                 int* values_out,
                                 int count,
                                 int begin_bit,
                                 int end_bit,
                                 cudaStream_t stream)
{
  return cub::DeviceRadixSort::SortPairs(storage,
                                         storage_bytes,
                                         keys_in,
                                         keys_out,
                                         values_in,
                                         values_out,
                                         count,
                                         begin_bit,
                                         end_bit,
                                         stream);
}

}  // namespace cudf::detail
