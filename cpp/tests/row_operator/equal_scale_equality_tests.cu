/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/dictionary/encode.hpp>
#include <cudf/table/equality.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/limits>
#include <thrust/transform.h>

#include <vector>

namespace {

template <typename Comparator>
struct compare_row_pairs {
  Comparator equal;
  cudf::size_type rows;

  __device__ bool operator()(cudf::size_type pair) const { return equal(pair / rows, pair % rows); }
};

template <typename Comparator>
struct compare_corresponding_rows {
  Comparator equal;

  __device__ bool operator()(cudf::size_type row) const
  {
    return equal(cudf::detail::row::lhs_index_type{row}, cudf::detail::row::rhs_index_type{row});
  }
};

template <typename Comparator>
void expect_self_equality(cudf::table_view input,
                          std::vector<bool> const& expected,
                          cudf::null_equality nulls,
                          Comparator comparator,
                          cuda::stream_ref stream,
                          cudf::memory_resources mr)
{
  auto const rows = input.num_rows();
  auto output     = cudf::make_numeric_column(cudf::data_type{cudf::type_id::BOOL8},
                                          rows * rows,
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr.get_output_mr());
  auto const self =
    cudf::detail::row::equality::self_comparator{input, stream, mr.get_temporary_mr()};
  auto const check_nulls = cudf::nullate::DYNAMIC{cudf::has_nested_nulls(input)};
  // Comparing different rows exercises unequal values as well as duplicate representations.
  auto const compare = [&](auto equal) {
    thrust::transform(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                      cuda::counting_iterator<cudf::size_type>{0},
                      cuda::counting_iterator<cudf::size_type>{rows * rows},
                      output->mutable_view().data<bool>(),
                      compare_row_pairs{equal, rows});
  };
  if (cudf::has_nested_columns(input)) {
    compare(self.equal_to<true>(check_nulls, nulls, comparator));
  } else {
    compare(self.equal_to<false>(check_nulls, nulls, comparator));
  }
  auto const expected_column =
    cudf::test::fixed_width_column_wrapper<bool>{expected.begin(), expected.end(), stream, mr};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    expected_column, output->view(), cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
}

template <typename Rep>
std::vector<bool> expected_equality(std::vector<Rep> const& values,
                                    std::vector<int> const& validity,
                                    cudf::null_equality nulls)
{
  std::vector<bool> expected;
  for (std::size_t lhs = 0; lhs < values.size(); ++lhs) {
    for (std::size_t rhs = 0; rhs < values.size(); ++rhs) {
      bool const lhs_valid = validity.empty() || validity[lhs];
      bool const rhs_valid = validity.empty() || validity[rhs];
      expected.push_back(lhs_valid && rhs_valid
                           ? values[lhs] == values[rhs]
                           : nulls == cudf::null_equality::EQUAL && !lhs_valid && !rhs_valid);
    }
  }
  return expected;
}

struct scale_observing_equality {
  numeric::scale_type expected_scale;

  template <typename T>
  __device__ bool operator()(T lhs, T rhs) const
  {
    if constexpr (cudf::is_fixed_point<T>()) {
      return lhs.scale() == expected_scale && rhs.scale() == expected_scale && lhs == rhs;
    } else {
      // Passing storage values to a custom comparator would change its contract.
      (void)lhs;
      (void)rhs;
      return false;
    }
  }
};

template <typename T>
struct EqualScaleEqualityTest : cudf::test::BaseFixtureWithHarness {};
TYPED_TEST_SUITE(EqualScaleEqualityTest, cudf::test::FixedPointTypes);

TYPED_TEST(EqualScaleEqualityTest, SameScaleMatchesRepresentations)
{
  using Rep         = cudf::device_storage_type_t<TypeParam>;
  auto const stream = this->stream();
  auto const mr     = this->resources();
  std::vector<Rep> const values{
    cuda::std::numeric_limits<Rep>::lowest(), -7, 0, 42, cuda::std::numeric_limits<Rep>::max(), 0};
  for (auto scale : {cuda::std::numeric_limits<int32_t>::lowest(),
                     -3,
                     0,
                     7,
                     cuda::std::numeric_limits<int32_t>::max()}) {
    for (auto const& validity : {std::vector<int>{},
                                 std::vector<int>{1, 0, 1, 0, 1, 1},
                                 std::vector<int>(values.size(), 0)}) {
      auto const input = validity.empty()
                           ? cudf::test::fixed_point_column_wrapper<Rep>{values.begin(),
                                                                         values.end(),
                                                                         numeric::scale_type{scale},
                                                                         stream,
                                                                         mr}
                           : cudf::test::fixed_point_column_wrapper<Rep>{values.begin(),
                                                                         values.end(),
                                                                         validity.begin(),
                                                                         numeric::scale_type{scale},
                                                                         stream,
                                                                         mr};
      for (auto nulls : {cudf::null_equality::EQUAL, cudf::null_equality::UNEQUAL}) {
        auto const expected = expected_equality(values, validity, nulls);
        expect_self_equality(cudf::table_view{{input}},
                             expected,
                             nulls,
                             cudf::detail::row::equality::nan_equal_physical_equality_comparator{},
                             stream,
                             mr);
        expect_self_equality(cudf::table_view{{input}},
                             expected,
                             nulls,
                             cudf::detail::row::equality::physical_equality_comparator{},
                             stream,
                             mr);
        expect_self_equality(cudf::table_view{{input}},
                             expected,
                             nulls,
                             scale_observing_equality{numeric::scale_type{scale}},
                             stream,
                             mr);
      }
    }
  }
}

TYPED_TEST(EqualScaleEqualityTest, NestedAndDictionarySlicesPreserveScale)
{
  using Rep         = cudf::device_storage_type_t<TypeParam>;
  auto const stream = this->stream();
  auto const mr     = this->resources();
  auto input        = cudf::test::fixed_point_column_wrapper<Rep>{
    {99, 4, 5, 4, 7, 99}, {1, 1, 0, 1, 1, 1}, numeric::scale_type{-2}, stream, mr};
  auto dictionary =
    cudf::dictionary::encode(input, cudf::data_type{cudf::type_id::INT32}, stream, mr);
  auto structures = cudf::test::structs_column_wrapper{{input}, {}, stream, mr};
  auto offsets =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{{0, 1, 2, 3, 4, 5, 6}, stream, mr};
  auto lists = cudf::make_lists_column(
    6,
    offsets.release(),
    std::make_unique<cudf::column>(
      static_cast<cudf::column_view>(input), stream, mr.get_output_mr()),
    0,
    cuda::device_buffer<cudf::bitmask_type>{stream, mr.get_output_mr(), 0, cuda::no_init});
  for (auto view : {static_cast<cudf::column_view>(input),
                    dictionary->view(),
                    static_cast<cudf::column_view>(structures),
                    lists->view()}) {
    auto const sliced = cudf::slice(view, {1, 5}, stream).front();
    for (auto nulls : {cudf::null_equality::EQUAL, cudf::null_equality::UNEQUAL}) {
      auto const expected = expected_equality(std::vector<Rep>{4, 5, 4, 7}, {1, 0, 1, 1}, nulls);
      expect_self_equality(cudf::table_view{{sliced}},
                           expected,
                           nulls,
                           cudf::detail::row::equality::nan_equal_physical_equality_comparator{},
                           stream,
                           mr);
    }
  }
}

TYPED_TEST(EqualScaleEqualityTest, GeneralTwoTableEqualityKeepsScale)
{
  using Rep         = cudf::device_storage_type_t<TypeParam>;
  auto const stream = this->stream();
  auto const mr     = this->resources();
  auto const lhs =
    cudf::test::fixed_point_column_wrapper<Rep>{{10, 10, 20}, numeric::scale_type{-1}, stream, mr};
  auto const rhs = cudf::test::fixed_point_column_wrapper<Rep>{
    {100, 10, 200}, numeric::scale_type{-2}, stream, mr};
  auto const comparison = cudf::detail::row::equality::two_table_comparator{
    cudf::table_view{{lhs}}, cudf::table_view{{rhs}}, stream, mr.get_temporary_mr()};
  auto const equal = comparison.equal_to<false>(cudf::nullate::NO{});
  auto output = cudf::test::fixed_width_column_wrapper<bool>{{false, false, false}, stream, mr};
  thrust::transform(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                    cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator<cudf::size_type>{3},
                    cudf::mutable_column_view{output}.data<bool>(),
                    compare_corresponding_rows{equal});
  auto const expected =
    cudf::test::fixed_width_column_wrapper<bool>{{true, false, true}, stream, mr};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    expected, output, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
}

TYPED_TEST(EqualScaleEqualityTest, ExactTypeGateIncludesRecursiveDecimalScales)
{
  using Rep         = cudf::device_storage_type_t<TypeParam>;
  auto const stream = this->stream();
  auto const mr     = this->resources();
  auto lhs =
    cudf::test::fixed_point_column_wrapper<Rep>{{1, 2, 3}, numeric::scale_type{-1}, stream, mr};
  auto rhs =
    cudf::test::fixed_point_column_wrapper<Rep>{{10, 20, 30}, numeric::scale_type{-2}, stream, mr};
  auto left_dictionary =
    cudf::dictionary::encode(lhs, cudf::data_type{cudf::type_id::INT32}, stream, mr);
  auto right_dictionary =
    cudf::dictionary::encode(rhs, cudf::data_type{cudf::type_id::INT32}, stream, mr);
  auto left_struct     = cudf::test::structs_column_wrapper{{lhs}, {}, stream, mr};
  auto right_struct    = cudf::test::structs_column_wrapper{{rhs}, {}, stream, mr};
  auto const make_list = [&](cudf::column_view child) {
    auto offsets =
      cudf::test::fixed_width_column_wrapper<cudf::size_type>{{0, 1, 2, 3}, stream, mr};
    return cudf::make_lists_column(
      3,
      offsets.release(),
      std::make_unique<cudf::column>(child, stream, mr.get_output_mr()),
      0,
      cuda::device_buffer<cudf::bitmask_type>{stream, mr.get_output_mr(), 0, cuda::no_init});
  };
  auto left_list  = make_list(lhs);
  auto right_list = make_list(rhs);
  std::vector<cudf::column_view> const left{
    lhs, left_dictionary->view(), left_struct, left_list->view()};
  std::vector<cudf::column_view> const right{
    rhs, right_dictionary->view(), right_struct, right_list->view()};
  for (std::size_t i = 0; i < left.size(); ++i) {
    // These logical values are equal, but the production gate must reject their scales.
    EXPECT_FALSE(cudf::have_same_types(left[i], right[i]));
    EXPECT_FALSE(cudf::have_same_types(cudf::table_view{{left[i]}}, cudf::table_view{{right[i]}}));
    EXPECT_TRUE(cudf::have_same_types(left[i], left[i]));
    EXPECT_FALSE(cudf::tables_equal(cudf::table_view{{left[i]}},
                                    cudf::table_view{{right[i]}},
                                    cudf::null_equality::EQUAL,
                                    stream));
  }
}

}  // namespace
