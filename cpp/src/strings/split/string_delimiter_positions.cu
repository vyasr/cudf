/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "split.cuh"

namespace cudf::strings::detail {

rmm::device_uvector<int64_t> find_string_delimiter_positions(strings_column_view const& input,
                                                             cudf::string_view delimiter,
                                                             cuda::stream_ref stream)
{
  auto [first_offset, last_offset] = get_first_and_last_offset(input, stream);
  auto const chars_bytes           = last_offset - first_offset;
  auto delimiter_fn =
    string_delimiter_fn{delimiter, chars_bytes, input.chars_begin(stream) + first_offset};

  cudf::detail::device_scalar<int64_t> d_count(0, stream, cudf::get_current_device_resource_ref());
  if (chars_bytes > 0) {
    constexpr int64_t block_size         = 512;
    constexpr size_type bytes_per_thread = 4;
    auto const num_blocks                = util::div_rounding_up_safe(
      util::div_rounding_up_safe(chars_bytes, static_cast<int64_t>(bytes_per_thread)), block_size);
    count_delimiters_kernel<string_delimiter_fn, block_size, bytes_per_thread>
      <<<num_blocks, block_size, 0, stream.get()>>>(delimiter_fn, chars_bytes, d_count.data());
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  auto positions = rmm::device_uvector<int64_t>(d_count.value(stream), stream);
  cudf::detail::copy_if_async(cuda::counting_iterator<int64_t>{0},
                              cuda::counting_iterator<int64_t>{chars_bytes},
                              positions.begin(),
                              delimiter_fn,
                              stream);
  return positions;
}

rmm::device_uvector<int64_t> find_whitespace_delimiter_positions(strings_column_view const& input,
                                                                 cuda::stream_ref stream)
{
  auto [first_offset, last_offset] = get_first_and_last_offset(input, stream);
  auto const chars_bytes           = last_offset - first_offset;
  auto delimiter_fn =
    whitespace_delimiter_fn{chars_bytes, input.chars_begin(stream) + first_offset};

  cudf::detail::device_scalar<int64_t> d_count(0, stream, cudf::get_current_device_resource_ref());
  if (chars_bytes > 0) {
    constexpr int64_t block_size         = 512;
    constexpr size_type bytes_per_thread = 4;
    auto const num_blocks                = util::div_rounding_up_safe(
      util::div_rounding_up_safe(chars_bytes, static_cast<int64_t>(bytes_per_thread)), block_size);
    count_delimiters_kernel<whitespace_delimiter_fn, block_size, bytes_per_thread>
      <<<num_blocks, block_size, 0, stream.get()>>>(delimiter_fn, chars_bytes, d_count.data());
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  auto positions = rmm::device_uvector<int64_t>(d_count.value(stream), stream);
  cudf::detail::copy_if_async(cuda::counting_iterator<int64_t>{0},
                              cuda::counting_iterator<int64_t>{chars_bytes},
                              positions.begin(),
                              delimiter_fn,
                              stream);
  return positions;
}

}  // namespace cudf::strings::detail
