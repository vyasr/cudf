/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "hash_join.cuh"
#include "join_common_utils.hpp"
#include "join_kernels.cuh"

#include <cudf/ast/detail/transform.cuh>
#include <cudf/ast/nodes.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/optional.h>

#include <algorithm>
#include <limits>

namespace cudf {
namespace detail {

/**
 * @brief Computes the join operation between two tables and returns the
 * output indices of left and right table as a combined table
 *
 * @param left  Table of left columns to join
 * @param right Table of right  columns to join
 * tables have been flipped, meaning the output indices should also be flipped
 * @param JoinKind The type of join to be performed
 * @param compare_nulls Controls whether null join-key values should match or not.
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return Join output indices vector pair
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
get_conditional_join_indices(table_view const& left,
                             table_view const& right,
                             join_kind JoinKind,
                             ast::expression binary_pred,
                             null_equality compare_nulls,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
{
  // We can immediately filter out cases where the right table is empty. In
  // some cases, we return all the rows of the left table with a corresponding
  // null index for the right table; in others, we return an empty output.
  if (right.num_rows() == 0) {
    switch (JoinKind) {
      // Left, left anti, and full (which are effectively left because we are
      // guaranteed that left has more rows than right) all return a all the
      // row indices from left with a corresponding NULL from the right.
      case join_kind::LEFT_JOIN:
      case join_kind::LEFT_ANTI_JOIN:
      case join_kind::FULL_JOIN: return get_trivial_left_join_indices(left, stream);
      // Inner and left semi joins return empty output because no matches can exist.
      case join_kind::INNER_JOIN:
      case join_kind::LEFT_SEMI_JOIN:
        return std::make_pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                              std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
    }
  }

  // Prepare output column. Whether or not the output column is nullable is
  // determined by whether any of the columns in the input table are nullable.
  // If none of the input columns actually contain nulls, we can still use the
  // non-nullable version of the expression evaluation code path for
  // performance, so we capture that information as well.
  auto const nullable =
    std::any_of(left.begin(), left.end(), [](column_view c) { return c.nullable(); }) ||
    std::any_of(right.begin(), right.end(), [](column_view c) { return c.nullable(); });
  auto const has_nulls = nullable && (cudf::has_nulls(left) || cudf::has_nulls(right));

  auto const plan = ast::detail::ast_plan{binary_pred, left, right, has_nulls, stream, mr};
  CUDF_EXPECTS(plan.output_type().id() == type_id::BOOL8,
               "The expression must produce a boolean output.");

  auto left_table  = table_device_view::create(left, stream);
  auto right_table = table_device_view::create(right, stream);

  // Allocate storage for the counter used to get the size of the join output
  rmm::device_scalar<size_type> size(0, stream, mr);
  CHECK_CUDA(stream.value());
  constexpr int block_size{DEFAULT_JOIN_BLOCK_SIZE};

  int numBlocks{-1};
  if (has_nulls) {
      CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &numBlocks, compute_conditional_join_output_size<block_size, true>, block_size, plan.dev_plan.shmem_per_thread * block_size));
  } else {
      CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &numBlocks, compute_conditional_join_output_size<block_size, false>, block_size, plan.dev_plan.shmem_per_thread * block_size));
  }

  int dev_id{-1};
  CUDA_TRY(cudaGetDevice(&dev_id));

  int num_sms{-1};
  CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));


  detail::grid_1d config(left_table->num_rows(), block_size);
  auto const shmem_size_per_block = plan.dev_plan.shmem_per_thread * config.num_threads_per_block;

  // Determine number of output rows without actually building the output to simply
  // find what the size of the output will be.
  join_kind KernelJoinKind = JoinKind == join_kind::FULL_JOIN ? join_kind::LEFT_JOIN : JoinKind;
  if (has_nulls) {
    compute_conditional_join_output_size<block_size, true>
      <<<numBlocks * num_sms, block_size, plan.dev_plan.shmem_per_thread * block_size, stream.value()>>>(
        *left_table, *right_table, KernelJoinKind, compare_nulls, plan.dev_plan, size.data());
  } else {
    compute_conditional_join_output_size<block_size, false>
      <<<numBlocks * num_sms, block_size, plan.dev_plan.shmem_per_thread * block_size, stream.value()>>>(
        *left_table, *right_table, KernelJoinKind, compare_nulls, plan.dev_plan, size.data());
  }
  CHECK_CUDA(stream.value());

  size_type const join_size = size.value(stream);
  CUDF_EXPECTS(join_size < std::numeric_limits<size_type>::max(), "The result of this join is too large for a cudf column.");

  // If the output size will be zero, we can return immediately.
  if (join_size == 0) {
    return std::make_pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                          std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  rmm::device_scalar<size_type> write_index(0, stream);

  auto left_indices  = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);

  const auto& join_output_l = left_indices->data();
  const auto& join_output_r = right_indices->data();
  if (has_nulls) {
    conditional_join<block_size, DEFAULT_JOIN_CACHE_SIZE, true>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
        *left_table,
        *right_table,
        KernelJoinKind,
        compare_nulls,
        join_output_l,
        join_output_r,
        write_index.data(),
        plan.dev_plan,
        join_size);
  } else {
    conditional_join<block_size, DEFAULT_JOIN_CACHE_SIZE, false>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
        *left_table,
        *right_table,
        KernelJoinKind,
        compare_nulls,
        join_output_l,
        join_output_r,
        write_index.data(),
        plan.dev_plan,
        join_size);
  }

  CHECK_CUDA(stream.value());

  auto join_indices = std::make_pair(std::move(left_indices), std::move(right_indices));

  // For full joins, get the indices in the right table that were not joined to
  // by any row in the left table.
  if (JoinKind == join_kind::FULL_JOIN) {
    auto complement_indices = detail::get_left_join_indices_complement(
      join_indices.second, left.num_rows(), right.num_rows(), stream, mr);
    join_indices = detail::concatenate_vector_pairs(join_indices, complement_indices, stream);
  }
  return join_indices;
}

/**
 * @brief Gives an estimate of the size of the join output produced when
 * joining two tables together.
 *
 * @throw cudf::logic_error if JoinKind is not INNER_JOIN or LEFT_JOIN
 *
 * @param left The left hand table
 * @param right The right hand table
 * @param JoinKind The type of join to be performed
 * @param compare_nulls Controls whether null join-key values should match or not.
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return An estimate of the size of the output of the join operation
 */
size_type estimate_nested_loop_join_output_size(table_device_view left,
                                                table_device_view right,
                                                join_kind JoinKind,
                                                null_equality compare_nulls,
                                                rmm::cuda_stream_view stream)
{
  const size_type left_num_rows{left.num_rows()};
  const size_type right_num_rows{right.num_rows()};

  if (right_num_rows == 0) {
    // If the right table is empty, we know exactly how large the output
    // will be for the different types of joins and can return immediately
    switch (JoinKind) {
      // Inner join with an empty table will have no output
      case join_kind::INNER_JOIN: return 0;

      // Left join with an empty table will have an output of NULL rows
      // equal to the number of rows in the left table
      case join_kind::LEFT_JOIN: return left_num_rows;

      default: CUDF_FAIL("Unsupported join type");
    }
  }

  // Allocate storage for the counter used to get the size of the join output
  size_type h_size_estimate{0};
  rmm::device_scalar<size_type> size_estimate(0, stream);

  CHECK_CUDA(stream.value());

  constexpr int block_size{DEFAULT_JOIN_BLOCK_SIZE};
  detail::grid_1d config(left.num_rows(), block_size);
  //int numBlocks{-1};

  //CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
  //  &numBlocks, compute_nested_loop_join_output_size<block_size>, block_size, 0));

  //int dev_id{-1};
  //CUDA_TRY(cudaGetDevice(&dev_id));

  //int num_sms{-1};
  //CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  //size_estimate.set_value_zero(stream);

  row_equality equality{left, right, compare_nulls == null_equality::EQUAL};
  // Determine number of output rows without actually building the output to simply
  // find what the size of the output will be.
  compute_nested_loop_join_output_size<block_size>
    <<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
      left, right, JoinKind, equality, size_estimate.data());
  CHECK_CUDA(stream.value());

  h_size_estimate = size_estimate.value(stream);

  return h_size_estimate;
}

/**
 * @brief Computes the join operation between two tables and returns the
 * output indices of left and right table as a combined table
 *
 * @param left  Table of left columns to join
 * @param right Table of right  columns to join
 * @param flip_join_indices Flag that indicates whether the left and right
 * tables have been flipped, meaning the output indices should also be flipped
 * @param JoinKind The type of join to be performed
 * @param compare_nulls Controls whether null join-key values should match or not.
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return Join output indices vector pair
 */
std::pair<rmm::device_uvector<size_type>, rmm::device_uvector<size_type>>
get_base_nested_loop_join_indices(table_view const& left,
                                  table_view const& right,
                                  bool flip_join_indices,
                                  join_kind JoinKind,
                                  null_equality compare_nulls,
                                  rmm::cuda_stream_view stream)
{
  // The `right` table is always used for the inner loop. We want to use the smaller table
  // for the inner loop. Thus, if `left` is smaller than `right`, swap `left/right`.
  // if ((JoinKind == join_kind::INNER_JOIN) && (right.num_rows() > left.num_rows())) {
  //   return get_base_nested_loop_join_indices(right, left, true, JoinKind, compare_nulls, stream);
  // }

  auto left_table  = table_device_view::create(left, stream);
  auto right_table = table_device_view::create(right, stream);

  size_type estimated_size = estimate_nested_loop_join_output_size(
    *left_table, *right_table, JoinKind, compare_nulls, stream);

  // If the estimated output size is zero, return immediately
  if (estimated_size == 0) {
    return std::make_pair(rmm::device_uvector<size_type>{0, stream},
                          rmm::device_uvector<size_type>{0, stream});
  }

  // Because we are approximating the number of joined elements, our approximation
  // might be incorrect and we might have underestimated the number of joined elements.
  // As such we will need to de-allocate memory and re-allocate memory to ensure
  // that the final output is correct.
  size_type join_size{0};

  rmm::device_uvector<size_type> left_indices{0, stream};
  rmm::device_uvector<size_type> right_indices{0, stream};
  auto current_estimated_size = estimated_size;
  do {
    left_indices.resize(estimated_size, stream);
    right_indices.resize(estimated_size, stream);

    constexpr int block_size{DEFAULT_JOIN_BLOCK_SIZE};
    detail::grid_1d config(left_table->num_rows(), block_size);
    rmm::device_scalar<size_type> write_index(0, stream);

    row_equality equality{*left_table, *right_table, compare_nulls == null_equality::EQUAL};
    const auto& join_output_l = flip_join_indices ? right_indices.data() : left_indices.data();
    const auto& join_output_r = flip_join_indices ? left_indices.data() : right_indices.data();
    nested_loop_join<block_size, DEFAULT_JOIN_CACHE_SIZE>
      <<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(*left_table,
                                                                               *right_table,
                                                                               JoinKind,
                                                                               equality,
                                                                               join_output_l,
                                                                               join_output_r,
                                                                               write_index.data(),
                                                                               estimated_size);

    CHECK_CUDA(stream.value());

    join_size              = write_index.value(stream);
    current_estimated_size = estimated_size;
    estimated_size *= 2;
  } while ((current_estimated_size < join_size));

  left_indices.resize(join_size, stream);
  right_indices.resize(join_size, stream);
  return std::make_pair(std::move(left_indices), std::move(right_indices));
}

}  // namespace detail

}  // namespace cudf
