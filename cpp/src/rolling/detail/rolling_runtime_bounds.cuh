/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "rolling_utils.cuh"

#include <cuda/std/type_traits>

namespace cudf::detail::rolling {

template <direction Direction>
struct runtime_window_bounds {
  size_type const* bounds_or_labels;
  size_type const* group_offsets;
  size_type num_rows;
  size_type delta;

  [[nodiscard]] __device__ constexpr size_type operator()(size_type i) const
  {
    // One bounds type shares rolling kernels across policies. Optional pointers encode
    // the policy without another argument field; selection stays outside aggregation loops.
    if (group_offsets != nullptr) {
      return fixed_window_clamper<Direction, grouped>{{bounds_or_labels, group_offsets}, delta}(i);
    }
    if (bounds_or_labels != nullptr) { return bounds_or_labels[i]; }
    return fixed_window_clamper<Direction, ungrouped>{{num_rows}, delta}(i);
  }

  [[nodiscard]] __device__ constexpr size_type operator[](size_type i) const { return (*this)(i); }
};

// Preserve postprocessing's original iterator categories (including contiguous pointers).
// Only the common descriptor is sent to the shared rolling kernel.
template <typename Iterator, direction Direction>
struct runtime_window_input {
  Iterator iterator;
  runtime_window_bounds<Direction> bounds;
};

template <typename T>
inline constexpr bool is_runtime_window_input = false;

template <typename Iterator, direction Direction>
inline constexpr bool is_runtime_window_input<runtime_window_input<Iterator, Direction>> = true;

template <typename Iterator>
[[nodiscard]] auto postprocess_window_iterator(Iterator const& input)
{
  if constexpr (is_runtime_window_input<Iterator>) {
    return input.iterator;
  } else {
    return input;
  }
}

template <direction Direction, typename Grouping>
[[nodiscard]] auto make_fixed_window_bounds(size_type delta, Grouping grouper)
{
  static_assert(
    cuda::std::is_same_v<Grouping, ungrouped> || cuda::std::is_same_v<Grouping, grouped>,
    "Invalid grouping descriptor");
  auto const bounds = [&] {
    if constexpr (cuda::std::is_same_v<Grouping, ungrouped>) {
      return runtime_window_bounds<Direction>{nullptr, nullptr, grouper.num_rows, delta};
    } else {
      return runtime_window_bounds<Direction>{grouper.labels, grouper.offsets, 0, delta};
    }
  }();
  return runtime_window_input{make_clamped_window_iterator<Direction>(delta, grouper), bounds};
}

template <direction Direction>
[[nodiscard]] auto make_variable_window_bounds(size_type const* bounds)
{
  return runtime_window_input{bounds, runtime_window_bounds<Direction>{bounds, nullptr, 0, 0}};
}

using preceding_window_bounds = runtime_window_bounds<direction::PRECEDING>;
using following_window_bounds = runtime_window_bounds<direction::FOLLOWING>;

}  // namespace cudf::detail::rolling
