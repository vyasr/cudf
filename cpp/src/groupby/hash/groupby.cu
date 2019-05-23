/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf.h>
#include <bitmask.hpp>
#include <bitmask/bit_mask.cuh>
#include <copying.hpp>
#include <groupby.hpp>
#include <hash/concurrent_unordered_map.cuh>
#include <string/nvcategory_util.hpp>
#include <table.hpp>
#include <table/device_table.cuh>
#include <table/device_table_row_operators.cuh>
#include <utilities/cuda_utils.hpp>
#include <utilities/device_atomics.cuh>
#include <utilities/release_assert.cuh>
#include <utilities/type_dispatcher.hpp>
#include "groupby.hpp"
#include "groupby_kernels.cuh"
#include "type_info.hpp"

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/fill.h>
#include <algorithm>
#include <type_traits>
#include <vector>

namespace cudf {
namespace groupby {
namespace hash {
namespace {
/**---------------------------------------------------------------------------*
 * @brief Verifies the requested aggregation is valid for the type of the value
 * column.
 *
 * Given a table of values and a set of operators, verifies that `ops[i]` is
 * valid to perform on `column[i]`.
 *
 * @throw cudf::logic_error if an invalid combination of value type and operator
 * is requested.
 *
 * @param values The table of columns
 * @param ops The aggregation operators
 *---------------------------------------------------------------------------**/
void verify_operators(table const& values, std::vector<operators> const& ops) {
  CUDF_EXPECTS(static_cast<gdf_size_type>(ops.size()) == values.num_columns(),
               "Size mismatch between ops and value columns");
  for (gdf_size_type i = 0; i < values.num_columns(); ++i) {
    // TODO Add more checks here, i.e., can't compute sum of non-arithemtic
    // types
    if ((ops[i] == SUM) and
        (values.get_column(i)->dtype == GDF_STRING_CATEGORY)) {
      CUDF_FAIL(
          "Cannot compute SUM aggregation of GDF_STRING_CATEGORY column.");
    }
  }
}

/**---------------------------------------------------------------------------*
 * @brief Deteremines target gdf_dtypes to use for combinations of source
 * gdf_dtypes and aggregation operations.
 *
 * Given vectors of source gdf_dtypes and corresponding aggregation operations
 * to be performed on that type, returns a vector the gdf_dtypes to use to store
 * the result of the aggregation operations.
 *
 * @param source_dtypes The source types
 * @param op The aggregation operations
 * @return Target gdf_dtypes to use for the target aggregation columns
 *---------------------------------------------------------------------------**/
inline std::vector<gdf_dtype> target_dtypes(
    std::vector<gdf_dtype> const& source_dtypes,
    std::vector<operators> const& ops) {
  std::vector<gdf_dtype> output_dtypes(source_dtypes.size());

  std::transform(
      source_dtypes.begin(), source_dtypes.end(), ops.begin(),
      output_dtypes.begin(), [](gdf_dtype source_dtype, operators op) {
        gdf_dtype t =
            cudf::type_dispatcher(source_dtype, target_type_mapper{}, op);
        CUDF_EXPECTS(
            t != GDF_invalid,
            "Invalid combination of input type and aggregation operation.");
        return t;
      });

  return output_dtypes;
}

struct identity_initializer {
  template <typename T>
  T get_identity(operators op) {
    switch (op) {
      case SUM:
        return corresponding_functor_t<SUM>::identity<T>();
      case MIN:
        return corresponding_functor_t<MIN>::identity<T>();
      case MAX:
        return corresponding_functor_t<MAX>::identity<T>();
      case COUNT:
        return corresponding_functor_t<COUNT>::identity<T>();
      default:
        CUDF_FAIL("Invalid aggregation operation.");
    }
  }

  template <typename T>
  void operator()(gdf_column const& col, operators op,
                  cudaStream_t stream = 0) {
    T* typed_data = static_cast<T*>(col.data);
    thrust::fill(rmm::exec_policy(stream)->on(stream), typed_data,
                 typed_data + col.size, get_identity<T>(op));

    // For COUNT operator, initialize column's bitmask to be all valid
    if ((nullptr != col.valid) and (COUNT == op)) {
      CUDA_TRY(cudaMemsetAsync(
          col.valid, 0xff,
          sizeof(gdf_valid_type) * gdf_valid_allocation_size(col.size),
          stream));
    }
  }
};

/**---------------------------------------------------------------------------*
 * @brief Initializes each column in a table with a corresponding identity value
 * of an aggregation operation.
 *
 * The `i`th column will be initialized with the identity value of the `i`th
 * aggregation operation.
 *
 * @note The validity bitmask for the column corresponding to a COUNT operator
 * will be initialized to all valid.
 *
 * @param table The table of columns to initialize.
 * @param operators The aggregation operations whose identity values will be
 *used to initialize the columns.
 *---------------------------------------------------------------------------**/
void initialize_with_identity(cudf::table const& table,
                              std::vector<operators> const& ops,
                              cudaStream_t stream = 0) {
  // TODO: Initialize all the columns in a single kernel instead of invoking one
  // kernel per column
  for (gdf_size_type i = 0; i < table.num_columns(); ++i) {
    gdf_column const* col = table.get_column(i);
    cudf::type_dispatcher(col->dtype, identity_initializer{}, *col, ops[i]);
  }
}

/**---------------------------------------------------------------------------*
 * @brief Compacts any GDF_STRING_CATEGORY columns in the output keys or values.
 *
 * After the groupby operation, any GDF_STRING_CATEGORY column in either the
 * keys or values may reference only a subset of the strings in the original
 * input category. This function will create a new associated NVCategory object
 * for the output GDF_STRING_CATEGORY columns whose dictionary contains only the
 * strings referenced in the output result.
 *
 * @param[in] input_keys The set of input key columns
 * @param[in/out] output_keys The set of output key columns
 * @param[in] input_values The set of input value columns
 * @param[in/out] output_values The set of output value columns
 *---------------------------------------------------------------------------**/
void update_nvcategories(table const& input_keys, table& output_keys,
                         table const& input_values, table& output_values) {
  nvcategory_gather_table(input_keys, output_keys);
  nvcategory_gather_table(input_values, output_values);
}

template <bool keys_have_nulls, bool values_have_nulls, typename Map>
auto extract_results(table const& input_keys, table const& input_values,
                     device_table const& d_input_keys,
                     device_table const& d_input_values,
                     device_table const& d_sparse_output_values, Map* map,
                     cudaStream_t stream) {
  cudf::table output_keys{cudf::allocate_like(input_keys, stream)};
  cudf::table output_values{cudf::allocate_like(input_values, stream)};

  auto d_output_keys = device_table::create(output_keys);
  auto d_output_values = device_table::create(output_values);

  gdf_size_type* d_result_size{nullptr};
  RMM_TRY(RMM_ALLOC(&d_result_size, sizeof(gdf_size_type), stream));
  CUDA_TRY(cudaMemsetAsync(d_result_size, 0, sizeof(gdf_size_type), stream));

  cudf::util::cuda::grid_config_1d grid_params{input_keys.num_rows(), 256};

  extract_groupby_result<keys_have_nulls, values_have_nulls>
      <<<grid_params.num_blocks, grid_params.num_threads_per_block, 0,
         stream>>>(map, d_input_keys, *d_output_keys, d_sparse_output_values,
                   *d_output_values, d_result_size);

  CHECK_STREAM(stream);

  gdf_size_type result_size{-1};
  CUDA_TRY(cudaMemcpyAsync(&result_size, d_result_size, sizeof(gdf_size_type),
                           cudaMemcpyDeviceToHost, stream));

  // Update size and null count of output columns
  auto update_column = [result_size](gdf_column* col) {
    col->size = result_size;
    set_null_count(*col);
    return col;
  };

  std::transform(output_keys.begin(), output_keys.end(), output_keys.begin(),
                 update_column);
  std::transform(output_values.begin(), output_values.end(),
                 output_values.begin(), update_column);

  return std::make_tuple(output_keys, output_values);
}

template <bool keys_have_nulls, bool values_have_nulls>
auto compute_hash_groupby(cudf::table const& keys, cudf::table const& values,
                          std::vector<operators> const& ops, Options options,
                          cudaStream_t stream) {
  gdf_size_type constexpr unused_key{std::numeric_limits<gdf_size_type>::max()};
  gdf_size_type constexpr unused_value{
      std::numeric_limits<gdf_size_type>::max()};
  CUDF_EXPECTS(keys.num_rows() < unused_key, "Groupby input size too large.");

  // The exact output size is unknown a priori, therefore, use the input size as
  // an upper bound
  gdf_size_type const output_size_estimate{keys.num_rows()};

  cudf::table sparse_output_values{output_size_estimate,
                                   target_dtypes(column_dtypes(values), ops),
                                   values_have_nulls, false, stream};

  initialize_with_identity(sparse_output_values, ops, stream);

  auto const d_input_keys = device_table::create(keys);
  auto const d_input_values = device_table::create(values);
  auto d_sparse_output_values = device_table::create(sparse_output_values);
  rmm::device_vector<operators> d_ops(ops);

  // If we ignore null keys, then nulls are not equivalent
  bool const null_keys_are_equal{not options.ignore_null_keys};
  bool const skip_rows_with_nulls{keys_have_nulls and not null_keys_are_equal};

  row_hasher<keys_have_nulls> hasher{*d_input_keys};
  row_equality_comparator<keys_have_nulls> rows_equal{
      *d_input_keys, *d_input_keys, null_keys_are_equal};

  using map_type =
      concurrent_unordered_map<gdf_size_type, gdf_size_type, decltype(hasher),
                               decltype(rows_equal)>;

  auto map =
      std::make_unique<map_type>(compute_hash_table_size(keys.num_rows()),
                                 unused_key, unused_value, hasher, rows_equal);

  cudf::util::cuda::grid_config_1d grid_params{keys.num_rows(), 256};

  if (skip_rows_with_nulls) {
    auto row_bitmask{cudf::row_bitmask(keys, stream)};
    build_aggregation_table<true, values_have_nulls>
        <<<grid_params.num_blocks, grid_params.num_threads_per_block, 0,
           stream>>>(map.get(), *d_input_keys, *d_input_values,
                     *d_sparse_output_values, d_ops.data().get(),
                     row_bitmask.data().get());
  } else {
    build_aggregation_table<false, values_have_nulls>
        <<<grid_params.num_blocks, grid_params.num_threads_per_block, 0,
           stream>>>(map.get(), *d_input_keys, *d_input_values,
                     *d_sparse_output_values, d_ops.data().get(), nullptr);
  }
  CHECK_STREAM(stream);

  return extract_results<keys_have_nulls, values_have_nulls>(
      keys, values, *d_input_keys, *d_input_values, *d_sparse_output_values,
      map.get(), stream);
}

/**---------------------------------------------------------------------------*
 * @brief Returns instantiation of `compute_hash_groupby` based on presence of
 * null values in keys and values.
 *
 * @param keys The groupby key columns
 * @param values The groupby value columns
 * @return Specialized callable of compute_hash_groupby
 *---------------------------------------------------------------------------**/
auto groupby_null_specialization(table const& keys, table const& values) {
  if (cudf::has_nulls(keys)) {
    if (cudf::has_nulls(values)) {
      return compute_hash_groupby<true, true>;
    } else {
      return compute_hash_groupby<true, false>;
    }
  } else {
    if (cudf::has_nulls(values)) {
      return compute_hash_groupby<false, true>;
    } else {
      return compute_hash_groupby<false, false>;
    }
  }
}
}  // namespace
namespace detail {

std::tuple<cudf::table, cudf::table> groupby(cudf::table const& keys,
                                             cudf::table const& values,
                                             std::vector<operators> const& ops,
                                             Options options,
                                             cudaStream_t stream) {
  verify_operators(values, ops);

  auto compute_groupby = groupby_null_specialization(keys, values);

  cudf::table output_keys;
  cudf::table output_values;

  std::tie(output_keys, output_values) =
      compute_groupby(keys, values, ops, options, stream);

  update_nvcategories(keys, output_keys, values, output_values);

  return std::make_tuple(output_keys, output_values);
}
}  // namespace detail

std::tuple<cudf::table, cudf::table> groupby(cudf::table const& keys,
                                             cudf::table const& values,
                                             std::vector<operators> const& ops,
                                             Options options) {
  return detail::groupby(keys, values, ops, options);
}
}  // namespace hash
}  // namespace groupby
}  // namespace cudf
