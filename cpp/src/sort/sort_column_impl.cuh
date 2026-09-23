/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "sort.hpp"
#include "sort_radix.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/row_operator/common_utils.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_merge_sort.cuh>
#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/gather.h>
#include <thrust/transform.h>

#include <cstdint>
#include <type_traits>

namespace cudf {
namespace detail {

/**
 * @brief Extracts the first eight bytes of a string as an unsigned big-endian integer.
 *
 * Zero padding is order preserving. It can create a prefix tie between a short string and a
 * longer string containing zero bytes, but the full string comparator resolves every such tie.
 */
template <bool has_nulls>
struct string_prefix_extractor {
  __device__ uint64_t operator()(size_type row) const
  {
    if constexpr (has_nulls) {
      if (d_column.is_null(row)) { return 0; }
    }

    auto const string = d_column.element<string_view>(row);
    uint64_t prefix   = 0;
    for (size_type byte = 0; byte < 8; ++byte) {
      prefix <<= 8;
      if (byte < string.size_bytes()) { prefix |= static_cast<uint8_t>(string.data()[byte]); }
    }
    return prefix;
  }

  column_device_view const d_column;
};

/**
 * @brief String comparator accelerated by a contiguous array of eight-byte prefix keys.
 */
template <bool has_nulls>
struct string_prefix_comparator {
  __device__ bool operator()(size_type lhs, size_type rhs)
  {
    if constexpr (has_nulls) {
      bool const lhs_null{d_column.is_null(lhs)};
      bool const rhs_null{d_column.is_null(rhs)};
      if (lhs_null || rhs_null) {
        return null_compare(lhs_null, rhs_null, null_precedence) ==
               (ascending ? weak_ordering::LESS : weak_ordering::GREATER);
      }
    }

    auto const lhs_prefix = prefixes[lhs];
    auto const rhs_prefix = prefixes[rhs];
    if (lhs_prefix != rhs_prefix) {
      return ascending ? lhs_prefix < rhs_prefix : lhs_prefix > rhs_prefix;
    }

    auto const left_element  = d_column.element<string_view>(lhs);
    auto const right_element = d_column.element<string_view>(rhs);
    // Equal cached keys prove that the first eight bytes match whenever both values contain at
    // least eight bytes. Resume comparison at the first byte not represented by the key instead
    // of rescanning the known-equal prefix. Shorter values need the full comparison because zero
    // padding deliberately does not encode the distinction between a missing byte and '\0'.
    auto constexpr prefix_bytes = static_cast<size_type>(sizeof(uint64_t));
    if (left_element.size_bytes() >= prefix_bytes && right_element.size_bytes() >= prefix_bytes) {
      auto const left_suffix =
        string_view{left_element.data() + prefix_bytes, left_element.size_bytes() - prefix_bytes};
      auto const right_suffix =
        string_view{right_element.data() + prefix_bytes, right_element.size_bytes() - prefix_bytes};
      return ascending ? left_suffix < right_suffix : right_suffix < left_suffix;
    }
    return ascending ? left_element < right_element : right_element < left_element;
  }

  column_device_view const d_column;
  uint64_t const* prefixes;
  bool ascending;
  null_order null_precedence{};
};

/**
 * @brief Comparator functor needed for single column sort.
 *
 * @tparam Column element type.
 */
template <typename T>
struct simple_comparator {
  __device__ bool operator()(size_type lhs, size_type rhs)
  {
    if (has_nulls) {
      bool const lhs_null{d_column.is_null(lhs)};
      bool const rhs_null{d_column.is_null(rhs)};
      if (lhs_null || rhs_null) {
        return null_compare(lhs_null, rhs_null, null_precedence) ==
               (ascending ? weak_ordering::LESS : weak_ordering::GREATER);
      }
    }

    auto const left_element  = d_column.element<T>(lhs);
    auto const right_element = d_column.element<T>(rhs);
    return relational_compare(left_element, right_element) ==
           (ascending ? weak_ordering::LESS : weak_ordering::GREATER);
  }
  column_device_view const d_column;
  bool has_nulls;
  bool ascending;
  null_order null_precedence{};
};

template <sort_method method>
struct column_sorted_order_fn {
 private:
  template <typename Comparator>
  void merge_sort(mutable_column_view& indices, Comparator comp, cuda::stream_ref stream)
  {
    auto in_keys   = cuda::counting_iterator<cudf::size_type>{0};
    auto out_keys  = indices.begin<size_type>();
    auto tmp_bytes = std::size_t{0};
    if constexpr (method == sort_method::STABLE) {
      cub::DeviceMergeSort::StableSortKeysCopy(
        nullptr, tmp_bytes, in_keys, out_keys, indices.size(), comp, stream.get());
      auto tmp_stg = rmm::device_buffer(tmp_bytes, stream);
      cub::DeviceMergeSort::StableSortKeysCopy(
        tmp_stg.data(), tmp_bytes, in_keys, out_keys, indices.size(), comp, stream.get());
    } else {
      cub::DeviceMergeSort::SortKeysCopy(
        nullptr, tmp_bytes, in_keys, out_keys, indices.size(), comp, stream.get());
      auto tmp_stg = rmm::device_buffer(tmp_bytes, stream);
      cub::DeviceMergeSort::SortKeysCopy(
        tmp_stg.data(), tmp_bytes, in_keys, out_keys, indices.size(), comp, stream.get());
    }
  }

  template <bool has_nulls>
  void prefix_sorted_order_impl(column_view const& input,
                                column_device_view const& keys,
                                mutable_column_view& indices,
                                bool ascending,
                                null_order null_precedence,
                                cuda::stream_ref stream)
  {
    auto prefixes =
      rmm::device_uvector<uint64_t>(input.size(), stream, cudf::get_current_device_resource_ref());
    auto rows = cuda::counting_iterator<cudf::size_type>{0};
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      rows,
                      rows + input.size(),
                      prefixes.begin(),
                      string_prefix_extractor<has_nulls>{keys});

    auto comp =
      string_prefix_comparator<has_nulls>{keys, prefixes.data(), ascending, null_precedence};
    merge_sort(indices, comp, stream);
  }

  void prefix_sorted_order(column_view const& input,
                           mutable_column_view& indices,
                           bool ascending,
                           null_order null_precedence,
                           cuda::stream_ref stream)
  {
    auto keys = column_device_view::create(input, stream);
    if (input.has_nulls()) {
      prefix_sorted_order_impl<true>(input, *keys, indices, ascending, null_precedence, stream);
    } else {
      prefix_sorted_order_impl<false>(input, *keys, indices, ascending, null_precedence, stream);
    }
  }

 public:
  /**
   * @brief Sorts a single column with a relationally comparable type.
   *
   * This is used when a comparator is required.
   *
   * @param input Column to sort
   * @param indices Output sorted indices
   * @param ascending True if sort order is ascending
   * @param null_precedence How null rows are to be ordered
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  template <typename T>
  void sorted_order(column_view const& input,
                    mutable_column_view& indices,
                    bool ascending,
                    null_order null_precedence,
                    cuda::stream_ref stream)
  {
    if constexpr (std::is_same_v<T, string_view>) {
      prefix_sorted_order(input, indices, ascending, null_precedence, stream);
      return;
    }

    auto keys = column_device_view::create(input, stream);
    auto comp = simple_comparator<T>{*keys, input.has_nulls(), ascending, null_precedence};
    merge_sort(indices, comp, stream);
  }

  template <typename T>
    requires(cudf::is_relationally_comparable<T, T>() and not cudf::is_dictionary<T>())
  void operator()(column_view const& input,
                  mutable_column_view& indices,
                  bool ascending,
                  null_order null_precedence,
                  cuda::stream_ref stream)
  {
    sorted_order<T>(input, indices, ascending, null_precedence, stream);
  }

  template <typename T>
    requires(not cudf::is_relationally_comparable<T, T>())
  void operator()(column_view const&, mutable_column_view&, bool, null_order, cuda::stream_ref)
  {
    CUDF_FAIL("Column type must be relationally comparable");
  }

  template <typename T>
    requires(is_dictionary<T>())
  void operator()(column_view const& input,
                  mutable_column_view& indices,
                  bool ascending,
                  null_order null_precedence,
                  cuda::stream_ref stream)
  {
    auto const keys = dictionary_column_view(input).keys();
    // For the keys we do an arg-sort of arg-sort to get the rank and use that as a map
    // to sort the indices in rank order.
    // First, get sorted-order of just the keys (slow but expect keys.size <<< indices.size)
    auto temp_mr = cudf::get_current_device_resource_ref();
    auto ordered_indices =
      cudf::detail::sorted_order<method>(keys, order::ASCENDING, null_precedence, stream, temp_mr);
    // Now, sort the ordered indices to get their ordered positions (very fast integer sort)
    ordered_indices = cudf::detail::sorted_order<method>(
      ordered_indices->view(), order::ASCENDING, null_precedence, stream, temp_mr);
    // And use the result as a map over the dictionary indices
    auto map = ordered_indices->view().template data<size_type>();
    auto itr = cudf::detail::indexalator_factory::make_input_iterator(
      dictionary_column_view(input).indices());
    auto mapped_indices = rmm::device_uvector<size_type>(input.size(), stream);
    thrust::gather(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   itr,
                   itr + input.size(),
                   map,
                   mapped_indices.begin());

    // Finally, sort-order the dictionary indices using mapped values
    auto mapped_view = column_view(data_type{type_to_id<size_type>()},
                                   input.size(),
                                   mapped_indices.data(),
                                   input.null_mask(),
                                   input.null_count());
    // these should be very fast since they are sorting integers
    if (input.has_nulls()) {
      sorted_order<size_type>(mapped_view, indices, ascending, null_precedence, stream);
    } else {
      sorted_order_radix(mapped_view, indices, ascending, stream);
    }
  }
};

}  // namespace detail
}  // namespace cudf
