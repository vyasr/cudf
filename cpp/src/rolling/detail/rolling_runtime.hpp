/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "rolling_runtime_bounds.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <memory>

namespace cudf::detail {

// The definition is private to typed owners: common bounds types alone cannot deduplicate
// device code emitted by separate CUDA translation units.
// Owning only the launch keeps preprocessing/postprocessing independent of these type shards.
struct dispatch_runtime_rolling {
  template <typename InputType>
  std::unique_ptr<column> operator()(column_view const& input,
                                     column_view const& default_outputs,
                                     rolling::preceding_window_bounds preceding_window_begin,
                                     rolling::following_window_bounds following_window_begin,
                                     size_type min_periods,
                                     rolling_aggregation const& agg,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr);
};

}  // namespace cudf::detail
