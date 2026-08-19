/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>

#include <cuda/stream>

#include <rapidsmpf/cuda_stream.hpp>

#include <ranges>
#include <utility>

namespace cudf_streaming::detail {

inline rmm::cuda_stream_view as_rmm_stream_view(rmm::cuda_stream_view stream) { return stream; }

inline rmm::cuda_stream_view as_rmm_stream_view(cuda::stream_ref stream)
{
  return rmm::cuda_stream_view{stream.get()};
}

template <typename Range>
auto as_rmm_stream_view_range(Range&& streams)
{
  return std::forward<Range>(streams) |
         std::views::transform([](auto stream) { return as_rmm_stream_view(stream); });
}

template <typename Downstream, typename Upstream>
  requires(!std::ranges::range<Downstream> && !std::ranges::range<Upstream>)
void mpf_cuda_stream_join(Downstream downstream,
                          Upstream upstream,
                          rapidsmpf::CudaEvent* event = nullptr)
{
  rapidsmpf::cuda_stream_join(as_rmm_stream_view(downstream), as_rmm_stream_view(upstream), event);
}

template <typename Downstreams, typename Upstreams>
  requires(std::ranges::range<Downstreams> && std::ranges::range<Upstreams>)
void mpf_cuda_stream_join(Downstreams&& downstreams,
                          Upstreams&& upstreams,
                          rapidsmpf::CudaEvent* event = nullptr)
{
  auto downstream_views = as_rmm_stream_view_range(std::forward<Downstreams>(downstreams));
  auto upstream_views   = as_rmm_stream_view_range(std::forward<Upstreams>(upstreams));
  rapidsmpf::cuda_stream_join(downstream_views, upstream_views, event);
}

}  // namespace cudf_streaming::detail
