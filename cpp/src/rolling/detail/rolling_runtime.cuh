/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "rolling.cuh"

namespace cudf::detail {

template <typename InputType>
std::unique_ptr<column> dispatch_runtime_rolling::operator()(
  column_view const& input,
  column_view const& default_outputs,
  rolling::preceding_window_bounds preceding_window_begin,
  rolling::following_window_bounds following_window_begin,
  size_type min_periods,
  rolling_aggregation const& agg,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  return aggregation_dispatcher(agg.kind,
                                rolling_window_launcher<InputType>{},
                                input,
                                default_outputs,
                                preceding_window_begin,
                                following_window_begin,
                                min_periods,
                                agg,
                                stream,
                                mr);
}

}  // namespace cudf::detail

// Shard type ownership to share window policies without serializing all type compilation.
#define CUDF_INSTANTIATE_RUNTIME_ROLLING(T)                                                     \
  template std::unique_ptr<cudf::column> cudf::detail::dispatch_runtime_rolling::operator()<T>( \
    cudf::column_view const&,                                                                   \
    cudf::column_view const&,                                                                   \
    cudf::detail::rolling::preceding_window_bounds,                                             \
    cudf::detail::rolling::following_window_bounds,                                             \
    cudf::size_type,                                                                            \
    cudf::rolling_aggregation const&,                                                           \
    cuda::stream_ref,                                                                           \
    cudf::memory_resources);
