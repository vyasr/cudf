/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/sort/group_single_pass_reduction_util.cuh"
#include "reductions/nested_types_extrema_utils.cuh"

#include <rmm/device_uvector.hpp>

#include <thrust/gather.h>

namespace cudf::groupby::detail {

std::unique_ptr<column> group_argminmax_indices(column_view const& values,
                                                size_type num_groups,
                                                cudf::device_span<size_type const> group_labels,
                                                bool is_argmin,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
{
  auto result = make_fixed_width_column(
    data_type{type_id::INT32}, num_groups, mask_state::UNALLOCATED, stream, mr);
  if (values.is_empty()) { return result; }

  auto d_values_ptr = decltype(column_device_view::create(values, stream)){};
  if (!cudf::is_nested(values.type()) || values.has_nulls()) {
    d_values_ptr = column_device_view::create(values, stream);
  }

  auto const result_begin = result->mutable_view().begin<size_type>();
  if (cudf::is_nested(values.type())) {
    auto const binop_generator =
      cudf::reduction::detail::arg_minmax_binop_generator::create(values, is_argmin, stream);
    launch_argminmax_reduction(group_labels, binop_generator.binop(), result_begin, stream);
  } else {
    auto value_type = cudf::is_dictionary(values.type())
                        ? dictionary_column_view(values).keys().type()
                        : values.type();
    launch_argminmax_reduction(
      group_labels, value_type, *d_values_ptr, values.has_nulls(), is_argmin, result_begin, stream);
  }

  if (values.has_nulls()) {
    rmm::device_uvector<bool> validity(num_groups, stream);
    reduce_group_validity(group_labels, *d_values_ptr, validity, stream);
    auto [null_mask, null_count] =
      cudf::detail::valid_if(validity.begin(), validity.end(), cuda::std::identity{}, stream, mr);
    result->set_null_mask(std::move(null_mask), null_count);
  }
  return result;
}

std::unique_ptr<column> group_argminmax(column_view const& values,
                                        size_type num_groups,
                                        device_span<size_type const> group_labels,
                                        column_view const& key_sort_order,
                                        bool is_argmin,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  auto dispatch_type = cudf::is_dictionary(values.type())
                         ? dictionary_column_view(values).keys().type()
                         : values.type();
  auto indices       = is_argmin ? type_dispatcher(dispatch_type,
                                             group_reduction_dispatcher<aggregation::ARGMIN>{},
                                             values,
                                             num_groups,
                                             group_labels,
                                             stream,
                                             mr)
                                 : type_dispatcher(dispatch_type,
                                             group_reduction_dispatcher<aggregation::ARGMAX>{},
                                             values,
                                             num_groups,
                                             group_labels,
                                             stream,
                                             mr);

  // Convert group-sorted indices back to the original row order. Using Thrust rather than
  // cudf::gather lets both operations move the reduction's null mask without copying it.
  auto indices_view = indices->view();
  auto output       = rmm::device_uvector<size_type>(indices_view.size(), stream, mr);
  thrust::gather(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                 indices_view.begin<size_type>(),
                 indices_view.end<size_type>(),
                 key_sort_order.begin<size_type>(),
                 output.data());
  auto null_count = indices_view.null_count();
  auto null_mask  = indices->release().null_mask.release();
  return std::make_unique<column>(std::move(output), std::move(*null_mask), null_count);
}

}  // namespace cudf::groupby::detail
