/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "page_decode.cuh"

namespace cudf::io::parquet::detail {

#define CUDF_PARQUET_PAGE_STATE_ERROR_METHODS                                                  \
  inline __device__ void set_error_code(decode_error err)                                      \
  {                                                                                            \
    cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_block> ref{setup.error};     \
    ref.fetch_or(static_cast<kernel_error::value_type>(err), cuda::std::memory_order_relaxed); \
  }                                                                                            \
  inline __device__ void reset_error_code()                                                    \
  {                                                                                            \
    cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_block> ref{setup.error};     \
    ref.store(0, cuda::std::memory_order_release);                                             \
  }

struct compute_page_sizes_state {
  page_decode_setup_state setup;
  page_decode_stream_state stream;
  page_decode_nesting_state nesting;
  page_decode_progress_state progress;
  page_decode_output_state output_cvt;
  CUDF_PARQUET_PAGE_STATE_ERROR_METHODS
};

struct preprocess_levels_state {
  page_decode_setup_state setup;
  page_decode_stream_state stream;
  CUDF_PARQUET_PAGE_STATE_ERROR_METHODS
};

struct decode_split_page_data_state {
  page_decode_setup_state setup;
  page_decode_stream_state stream;
  page_decode_nesting_state nesting;
  page_decode_progress_state progress;
  page_decode_output_state output_cvt;
  CUDF_PARQUET_PAGE_STATE_ERROR_METHODS
};

struct decode_page_data_state {
  page_decode_setup_state setup;
  page_decode_stream_state stream;
  page_decode_nesting_state nesting;
  page_decode_progress_state progress;
  page_decode_output_state output_cvt;
  CUDF_PARQUET_PAGE_STATE_ERROR_METHODS
};

struct decode_page_data_generic_state {
  page_decode_setup_state setup;
  page_decode_stream_state stream;
  page_decode_nesting_state nesting;
  page_decode_progress_state progress;
  page_decode_output_state output_cvt;
  CUDF_PARQUET_PAGE_STATE_ERROR_METHODS
};

struct decode_delta_binary_state {
  page_decode_setup_state setup;
  page_decode_stream_state stream;
  page_decode_nesting_state nesting;
  page_decode_progress_state progress;
  page_decode_output_state output_cvt;
  CUDF_PARQUET_PAGE_STATE_ERROR_METHODS
};

struct decode_delta_byte_array_state {
  page_decode_setup_state setup;
  page_decode_stream_state stream;
  page_decode_nesting_state nesting;
  page_decode_progress_state progress;
  page_decode_output_state output_cvt;
  CUDF_PARQUET_PAGE_STATE_ERROR_METHODS
};

struct decode_delta_length_byte_array_state {
  page_decode_setup_state setup;
  page_decode_stream_state stream;
  page_decode_nesting_state nesting;
  page_decode_progress_state progress;
  page_decode_output_state output_cvt;
  CUDF_PARQUET_PAGE_STATE_ERROR_METHODS
};

struct compute_string_page_bounds_state {
  page_decode_setup_state setup;
  page_decode_stream_state stream;
  page_decode_nesting_state nesting;
  page_decode_progress_state progress;
  page_decode_output_state output_cvt;
  CUDF_PARQUET_PAGE_STATE_ERROR_METHODS
};

struct compute_delta_page_string_sizes_state {
  page_decode_setup_state setup;
  page_decode_stream_state stream;
  page_decode_output_state output_cvt;
  CUDF_PARQUET_PAGE_STATE_ERROR_METHODS
};

struct compute_delta_length_page_string_sizes_state {
  page_decode_setup_state setup;
  page_decode_stream_state stream;
  page_decode_output_state output_cvt;
  CUDF_PARQUET_PAGE_STATE_ERROR_METHODS
};

struct compute_page_string_sizes_state {
  page_decode_setup_state setup;
  page_decode_stream_state stream;
  page_decode_output_state output_cvt;
  CUDF_PARQUET_PAGE_STATE_ERROR_METHODS
};

struct preprocess_string_offsets_state {
  page_decode_setup_state setup;
  page_decode_stream_state stream;
  page_decode_progress_state progress;
  page_decode_output_state output_cvt;
  CUDF_PARQUET_PAGE_STATE_ERROR_METHODS
};

#undef CUDF_PARQUET_PAGE_STATE_ERROR_METHODS

}  // namespace cudf::io::parquet::detail
