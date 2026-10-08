/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/sort/group_single_pass_reduction_util.cuh"
#include "groupby/sort/group_validity.cuh"

namespace cudf::groupby::detail {

std::unique_ptr<column> group_nested_argminmax(column_view const& values,
                                               size_type num_groups,
                                               cudf::device_span<size_type const> group_labels,
                                               bool is_argmin,
                                               cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr)
{
  auto result = make_fixed_width_column(
    data_type{type_id::INT32}, num_groups, mask_state::UNALLOCATED, stream, mr);

  if (values.is_empty()) { return result; }

  auto const result_begin = result->mutable_view().begin<size_type>();
  auto const binop_generator =
    cudf::reduction::detail::arg_minmax_binop_generator::create(values, is_argmin, stream);
  launch_argminmax_reduction(group_labels, binop_generator.binop(), result_begin, stream);

  if (values.has_nulls()) {
    auto const d_values_ptr = column_device_view::create(values, stream);
    auto validity           = rmm::device_uvector<bool>(num_groups, stream);
    reduce_group_validity(group_labels, *d_values_ptr, validity.data(), stream);

    auto [null_mask, null_count] =
      cudf::detail::valid_if(validity.begin(), validity.end(), cuda::std::identity{}, stream, mr);
    result->set_null_mask(std::move(null_mask), null_count);
  }

  return result;
}

}  // namespace cudf::groupby::detail
