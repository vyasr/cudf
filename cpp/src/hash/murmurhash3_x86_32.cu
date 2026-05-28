/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "murmurhash_jit_planner.hpp"
#include "runtime/context.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/row_operator/preprocessed_table.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_for.cuh>
#include <cuda_runtime.h>
#include <thrust/tabulate.h>

#include <cudf_hash_fragments.hpp>
#include <rtcx.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace cudf {
namespace hashing {
namespace detail {

std::unique_ptr<column> murmurhash3_x86_32(table_view const& input,
                                           uint32_t seed,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  auto output = make_numeric_column(data_type(type_to_id<hash_value_type>()),
                                    input.num_rows(),
                                    mask_state::UNALLOCATED,
                                    stream,
                                    mr);

  // Return early if there's nothing to hash
  if (input.num_columns() == 0 || input.num_rows() == 0) { return output; }

  // LTO path: use pre-compiled fragment kernels if all types are supported
  {
    auto fragments = cudf::hashing::detail::plan_murmurhash_fragments(input);
    if (!fragments.empty()) {
      rtcx::AlgorithmPlanner planner{"cudf_murmurhash3_x86_32_jit_link_kernel"};
      for (auto& fragment : fragments) {
        planner.add_fragment(std::move(fragment));
      }
      auto launcher = planner.get_launcher(cudf::get_context().rtcx_cache());

      bool nullable           = has_nulls(input);
      auto const preprocessed = cudf::detail::row::hash::preprocessed_table::create(input, stream);
      cudf::table_device_view input_dv{*preprocessed};

      auto output_view = output->mutable_view();
      auto d_output    = mutable_column_device_view::create(output_view, stream);

      cudf::detail::grid_1d const launch_grid{input.num_rows(), 256};

      rtcx::cuda_dim3 const grid{static_cast<std::uint32_t>(launch_grid.num_blocks), 1, 1};
      rtcx::cuda_dim3 const block{
        static_cast<std::uint32_t>(launch_grid.num_threads_per_block), 1, 1};

      auto output_dv = *d_output;
      using kernel_sig =
        void(cudf::mutable_column_device_view, std::uint32_t, cudf::table_device_view, bool);
      launcher.dispatch<kernel_sig>(
        stream.value(), grid, block, 0, output_dv, seed, input_dv, nullable);

      return output;
    }
  }

  bool const nullable   = has_nulls(input);
  auto const row_hasher = cudf::detail::row::hash::row_hasher(input, stream);
  auto output_view      = output->mutable_view();

  // Compute the hash value for each row
  auto const output_begin = output_view.begin<hash_value_type>();
  auto const hasher       = row_hasher.device_hasher<MurmurHash3_x86_32>(nullable, seed);
  // thrust::tabulate is slow here, see NVIDIA/cccl#9070
  CUDF_CUDA_TRY(cub::DeviceFor::Bulk(
    input.num_rows(),
    [output_begin, hasher] __device__(size_type i) mutable { output_begin[i] = hasher(i); },
    stream.value()));

  return output;
}

}  // namespace detail

std::unique_ptr<column> murmurhash3_x86_32(table_view const& input,
                                           uint32_t seed,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::murmurhash3_x86_32(input, seed, stream, mr);
}

}  // namespace hashing
}  // namespace cudf
