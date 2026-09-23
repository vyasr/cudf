/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/sorting.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <memory>
#include <string>

namespace {

std::unique_ptr<cudf::column> make_input(cudf::size_type num_rows,
                                         cudf::size_type min_width,
                                         cudf::size_type max_width,
                                         std::string const& workload,
                                         unsigned seed = 1)
{
  if (workload == "duplicates") {
    data_profile const profile =
      data_profile_builder().no_validity().cardinality(64).avg_run_length(1).distribution(
        cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
    return create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile, seed);
  }

  if (workload == "shared_prefix") {
    constexpr cudf::size_type suffix_width = 8;
    auto const prefix_width                = max_width - suffix_width;
    data_profile const prefix_profile =
      data_profile_builder()
        .no_validity()
        .cardinality(1)
        .distribution(cudf::type_id::STRING, distribution_id::UNIFORM, prefix_width, prefix_width)
        .string_char_range('a', 'a');
    data_profile const suffix_profile =
      data_profile_builder()
        .no_validity()
        .cardinality(0)
        .avg_run_length(1)
        .distribution(cudf::type_id::STRING, distribution_id::UNIFORM, suffix_width, suffix_width)
        .string_char_range(' ', '~');
    auto const prefix =
      create_random_column(cudf::type_id::STRING, row_count{num_rows}, prefix_profile, seed);
    auto const suffix =
      create_random_column(cudf::type_id::STRING, row_count{num_rows}, suffix_profile, seed + 1);
    return cudf::strings::concatenate(cudf::table_view{{prefix->view(), suffix->view()}});
  }

  if (workload == "normal") {
    data_profile const profile = data_profile_builder().distribution(
      cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
    return create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile, seed);
  }

  data_profile const profile =
    data_profile_builder().no_validity().cardinality(0).avg_run_length(1).distribution(
      cudf::type_id::STRING, distribution_id::UNIFORM, min_width, max_width);
  return create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile, seed);
}

std::unique_ptr<cudf::column> make_cardinality_input(cudf::size_type num_rows,
                                                     cudf::size_type max_width,
                                                     cudf::size_type cardinality,
                                                     unsigned seed)
{
  data_profile const profile =
    data_profile_builder()
      .no_validity()
      .cardinality(cardinality)
      .avg_run_length(1)
      .distribution(cudf::type_id::STRING, distribution_id::UNIFORM, 0, max_width);
  return create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile, seed);
}

std::unique_ptr<cudf::column> make_prefix_input(cudf::size_type num_rows,
                                                cudf::size_type prefix_width,
                                                unsigned seed)
{
  constexpr cudf::size_type suffix_width = 32;
  data_profile const prefix_profile =
    data_profile_builder()
      .no_validity()
      .cardinality(1)
      .distribution(cudf::type_id::STRING, distribution_id::UNIFORM, prefix_width, prefix_width)
      .string_char_range('a', 'a');
  data_profile const suffix_profile =
    data_profile_builder()
      .no_validity()
      .cardinality(0)
      .avg_run_length(1)
      .distribution(cudf::type_id::STRING, distribution_id::UNIFORM, 0, suffix_width)
      .string_char_range(' ', '~');
  auto const prefix =
    create_random_column(cudf::type_id::STRING, row_count{num_rows}, prefix_profile, seed);
  auto const suffix =
    create_random_column(cudf::type_id::STRING, row_count{num_rows}, suffix_profile, seed + 1);
  return cudf::strings::concatenate(cudf::table_view{{prefix->view(), suffix->view()}});
}

std::unique_ptr<cudf::column> make_length_input(cudf::size_type num_rows,
                                                std::string const& length_profile,
                                                double null_probability,
                                                unsigned seed)
{
  auto min_width    = cudf::size_type{0};
  auto max_width    = cudf::size_type{32};
  auto distribution = distribution_id::UNIFORM;
  if (length_profile == "fixed_8") {
    min_width = max_width = 8;
  } else if (length_profile == "fixed_64") {
    min_width = max_width = 64;
  } else if (length_profile == "variable_128") {
    max_width = 128;
  } else if (length_profile == "normal_128") {
    max_width    = 128;
    distribution = distribution_id::NORMAL;
  }

  data_profile const profile =
    data_profile_builder()
      .null_probability(null_probability)
      .cardinality(0)
      .avg_run_length(1)
      .distribution(cudf::type_id::STRING, distribution, min_width, max_width);
  return create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile, seed);
}

void run_sort_benchmark(nvbench::state& state, std::unique_ptr<cudf::column> const& input)
{
  auto const bytes = input->alloc_size();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.add_global_memory_reads<nvbench::int8_t>(bytes);
  state.add_global_memory_writes<nvbench::int8_t>(bytes);

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { cudf::sort(cudf::table_view{{input->view()}}); });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

}  // namespace

static void bench_sort_strings(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const workload  = state.get_string("workload");

  run_sort_benchmark(state, make_input(num_rows, min_width, max_width, workload));
}

static void bench_sort_strings_cardinality(nvbench::state& state)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const max_width   = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const seed        = static_cast<unsigned>(state.get_int64("seed"));
  run_sort_benchmark(state, make_cardinality_input(num_rows, max_width, cardinality, seed));
}

static void bench_sort_strings_prefix(nvbench::state& state)
{
  auto const num_rows     = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const prefix_width = static_cast<cudf::size_type>(state.get_int64("prefix_width"));
  auto const seed         = static_cast<unsigned>(state.get_int64("seed"));
  run_sort_benchmark(state, make_prefix_input(num_rows, prefix_width, seed));
}

static void bench_sort_strings_length(nvbench::state& state)
{
  auto const num_rows       = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const length_profile = state.get_string("length_profile");
  auto const null_percent   = static_cast<double>(state.get_int64("null_percent"));
  auto const seed           = static_cast<unsigned>(state.get_int64("seed"));
  run_sort_benchmark(state,
                     make_length_input(num_rows, length_profile, null_percent / 100.0, seed));
}

NVBENCH_BENCH(bench_sort_strings)
  .set_name("sort_strings")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_string_axis("workload", {"normal", "duplicates", "shared_prefix", "variable"});

// Measures the `sorted_order` fast-path case: a single strings column with no nulls
static void bench_sorted_order_strings(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));

  data_profile const profile =
    data_profile_builder()
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width)
      .no_validity();

  auto const table = create_random_table({cudf::type_id::STRING}, row_count{num_rows}, profile);
  auto const bytes = table->alloc_size();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.add_global_memory_reads<nvbench::int8_t>(bytes);
  state.add_global_memory_writes<cudf::size_type>(num_rows);

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { cudf::sorted_order(table->view()); });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_sorted_order_strings)
  .set_name("sorted_order_strings")
  .add_int64_axis("min_width", {1})
  .add_int64_axis("max_width", {8, 32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152, 16777216});

// Measures the multi-column (lexicographic row comparator) strings path
static void bench_sorted_order_strings_multi(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols  = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));

  data_profile const profile =
    data_profile_builder()
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, 1, max_width)
      .no_validity();

  auto const table = create_random_table(
    cycle_dtypes({cudf::type_id::STRING}, num_cols), row_count{num_rows}, profile);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.add_global_memory_reads<nvbench::int8_t>(table->alloc_size());
  state.add_global_memory_writes<cudf::size_type>(num_rows);

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { cudf::sorted_order(table->view()); });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_sorted_order_strings_multi)
  .set_name("sorted_order_strings_multi")
  .add_int64_axis("max_width", {8, 32, 64})
  .add_int64_axis("num_cols", {2, 4})
  .add_int64_axis("num_rows", {262144, 2097152});

// These targeted families exercise cardinality, common-prefix, and string-length sensitivity.
// Two seeds guard against tuning conclusions to a particular generated input.
NVBENCH_BENCH(bench_sort_strings_cardinality)
  .set_name("sort_strings_cardinality")
  .add_int64_axis("num_rows", {262144, 2097152})
  .add_int64_axis("max_width", {32, 128})
  .add_int64_axis("cardinality", {1, 4, 16, 64, 1024})
  .add_int64_axis("seed", {1, 17});

NVBENCH_BENCH(bench_sort_strings_prefix)
  .set_name("sort_strings_prefix")
  .add_int64_axis("num_rows", {262144, 2097152})
  .add_int64_axis("prefix_width", {0, 8, 24, 64, 128})
  .add_int64_axis("seed", {1, 17});

NVBENCH_BENCH(bench_sort_strings_length)
  .set_name("sort_strings_length")
  .add_int64_axis("num_rows", {262144, 2097152, 16777216})
  .add_string_axis("length_profile",
                   {"fixed_8", "fixed_64", "variable_32", "variable_128", "normal_128"})
  .add_int64_axis("null_percent", {0, 10})
  .add_int64_axis("seed", {1, 17});
