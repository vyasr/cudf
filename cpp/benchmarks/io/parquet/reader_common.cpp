/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reader_common.hpp"

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>

#include <cudf/io/parquet.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <nvbench/nvbench.cuh>

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace {

constexpr char const* prepass_env_var = "LIBCUDF_PARQUET_LEVEL_PREPASS";

/**
 * @brief Sets an environment variable for the duration of a scope, restoring it after.
 */
class scoped_env_var {
 public:
  scoped_env_var(char const* name, std::string const& value) : name_{name}
  {
    if (auto const* previous = std::getenv(name_); previous != nullptr) {
      previous_value_ = std::string{previous};
    }
    setenv(name_, value.c_str(), 1);
  }

  scoped_env_var(scoped_env_var const&)            = delete;
  scoped_env_var& operator=(scoped_env_var const&) = delete;
  scoped_env_var(scoped_env_var&&)                 = delete;
  scoped_env_var& operator=(scoped_env_var&&)      = delete;

  ~scoped_env_var()
  {
    if (previous_value_.has_value()) {
      setenv(name_, previous_value_->c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }

 private:
  char const* name_;
  std::optional<std::string> previous_value_;
};

}  // namespace

std::optional<double> null_probability_from_percent(int64_t null_percent)
{
  if (null_percent < 0) { return std::nullopt; }
  CUDF_EXPECTS(null_percent <= 100, "null_percent must be -1 or in [0, 100]");
  return static_cast<double>(null_percent) / 100.0;
}

std::optional<std::string> prepass_selector_for_mode(std::string_view mode)
{
  if (mode == "default") { return std::nullopt; }
  if (mode == "legacy") { return std::string{"0"}; }
  if (mode == "all") { return std::string{"0x1ff"}; }
  // Families plus one experimental probe, matching the bits in parquet_gpu.hpp. Names for
  // retired probes are removed rather than left pointing at a value the reader now rejects,
  // so a stale benchmark invocation fails where it is written instead of silently measuring
  // the default.
  if (mode == "list_bar") { return std::string{"0x81ff"}; }
  if (mode == "skip_shadow") { return std::string{"0x101ff"}; }
  if (mode == "warp_fused") { return std::string{"0x401ff"}; }
  if (mode == "warp_narrow") { return std::string{"0xc01ff"}; }
  if (mode == "warp_wide") { return std::string{"0x1001ff"}; }
  if (mode == "probes") { return std::string{"0x1d81ff"}; }
  if (mode.starts_with("0x")) { return std::string{mode}; }
  CUDF_FAIL("Unsupported prepass_mode: " + std::string{mode});
}

void parquet_read_common(cudf::size_type num_rows_to_read,
                         cudf::size_type num_cols_to_read,
                         cuio_source_sink_pair& source_sink,
                         nvbench::state& state)
{
  auto const data_size = static_cast<size_t>(state.get_int64("data_size"));
  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(source_sink.make_source_info());

  // `read_parquet` constructs a fresh reader per call and the reader reads this
  // variable in its constructor, so the selector can be varied per benchmark state
  // rather than per process. That keeps both arms of an A/B in one process, sharing
  // the input file, the allocator and the clock state.
  auto const prepass_selector = prepass_selector_for_mode(state.get_string("prepass_mode"));
  auto const prepass_guard    = prepass_selector.has_value() ? std::make_unique<scoped_env_var>(
                                                              prepass_env_var, *prepass_selector)
                                                             : nullptr;

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      drop_page_cache_if_enabled(read_opts.get_source().filepaths());

      timer.start();
      auto const result = cudf::io::read_parquet(read_opts);
      timer.stop();

      CUDF_EXPECTS(result.tbl->num_columns() == num_cols_to_read, "Unexpected number of columns");
      CUDF_EXPECTS(result.tbl->num_rows() == num_rows_to_read, "Unexpected number of rows");
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}
