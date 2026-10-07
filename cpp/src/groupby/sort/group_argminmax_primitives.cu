/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "group_argminmax_impl.cuh"

#include <cudf/detail/utilities/element_argminmax.cuh>
#include <cudf/utilities/type_dispatcher.hpp>

namespace cudf::groupby::detail {

#define INSTANTIATE_ARGMINMAX(T)                                                         \
  template void launch_argminmax_reduction(cudf::device_span<cudf::size_type const>,     \
                                           cudf::detail::element_argminmax_fn<T> const&, \
                                           cudf::size_type*,                             \
                                           cuda::stream_ref);

INSTANTIATE_ARGMINMAX(int8_t)
INSTANTIATE_ARGMINMAX(int16_t)
INSTANTIATE_ARGMINMAX(int32_t)
INSTANTIATE_ARGMINMAX(int64_t)
INSTANTIATE_ARGMINMAX(uint8_t)
INSTANTIATE_ARGMINMAX(uint16_t)
INSTANTIATE_ARGMINMAX(uint32_t)
INSTANTIATE_ARGMINMAX(uint64_t)
INSTANTIATE_ARGMINMAX(float)
INSTANTIATE_ARGMINMAX(double)
INSTANTIATE_ARGMINMAX(bool)
INSTANTIATE_ARGMINMAX(cudf::timestamp_D)
INSTANTIATE_ARGMINMAX(cudf::timestamp_s)
INSTANTIATE_ARGMINMAX(cudf::timestamp_ms)
INSTANTIATE_ARGMINMAX(cudf::timestamp_us)
INSTANTIATE_ARGMINMAX(cudf::timestamp_ns)
INSTANTIATE_ARGMINMAX(cudf::duration_D)
INSTANTIATE_ARGMINMAX(cudf::duration_s)
INSTANTIATE_ARGMINMAX(cudf::duration_ms)
INSTANTIATE_ARGMINMAX(cudf::duration_us)
INSTANTIATE_ARGMINMAX(cudf::duration_ns)
INSTANTIATE_ARGMINMAX(numeric::decimal32)
INSTANTIATE_ARGMINMAX(numeric::decimal64)
INSTANTIATE_ARGMINMAX(numeric::decimal128)
INSTANTIATE_ARGMINMAX(cudf::dictionary32)
INSTANTIATE_ARGMINMAX(cudf::string_view)

#undef INSTANTIATE_ARGMINMAX

}  // namespace cudf::groupby::detail
