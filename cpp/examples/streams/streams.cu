/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cudf/binaryop.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/join.hpp>
#include <cudf/sorting.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>

#include <iostream>

struct random_generator {
  float min, max;

  random_generator(float _min = 0.f, float _max = 1.f) : min{_min}, max{_max} {};

  __device__ float operator()(const unsigned int n) const
  {
    thrust::default_random_engine rng{};
    thrust::uniform_real_distribution<float> dist(min, max);
    rng.discard(n);
    return dist(rng);
  }
};

auto make_random_numeric_column(cudf::size_type num_rows)
{
  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, num_rows);
  // Make a column_device_view
  auto d_col = cudf::mutable_column_device_view::create(col->mutable_view());

  thrust::counting_iterator<unsigned int> index_sequence_begin(0);

  thrust::transform(rmm::exec_policy(cudf::get_default_stream()),
                    index_sequence_begin,
                    index_sequence_begin + num_rows,
                    d_col->begin<int32_t>(),
                    random_generator{1.f, 2.f});
  return col;
}

int main(int argc, char const** argv)
{
  // Create a cudf column and fill it with random integers
  cudf::size_type num_rows = 1000000000;
  cudf::size_type num_reps = 100;

  for (int i = 0; i < num_reps; ++i) {
    auto col1   = make_random_numeric_column(num_rows);
    auto col2   = make_random_numeric_column(num_rows);
    auto stream = rmm::cuda_stream{};

    auto output = cudf::binary_operation(
      col1->view(), col2->view(), cudf::binary_operator::ADD, col1->type(), stream);
  }
}
