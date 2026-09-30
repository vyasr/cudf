/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <charconv>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace cudf::detail {

/**
 * @brief Experimental single-column string sorting implementations.
 */
enum class string_sort_algorithm { PREFIX, SEGMENTED, SEGMENTED_RLE };

enum class segmented_rle_policy { NEVER, ALWAYS, ADAPTIVE };

struct segmented_string_sort_config {
  int bytes_per_pass{6};
  int radix_percent{100};
  int max_radix_passes{4};
  bool known_prefix{false};
  int finish_threshold{32};
  segmented_rle_policy rle_policy{segmented_rle_policy::NEVER};
  int rle_min_run_length{32};
  int rle_min_coverage_percent{25};
  int rle_min_equal_percent{10};
  bool trace{false};
  bool compact_radix{false};
};

[[nodiscard]] inline int parse_integer_setting(char const* value,
                                               int minimum,
                                               int maximum,
                                               int fallback)
{
  if (value == nullptr) { return fallback; }
  auto const text = std::string_view{value};
  if (text.empty()) { return fallback; }
  auto parsed             = int{};
  auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), parsed);
  return error == std::errc{} && end == text.data() + text.size() && parsed >= minimum &&
             parsed <= maximum
           ? parsed
           : fallback;
}

[[nodiscard]] inline int parse_bytes_per_pass(char const* value)
{
  if (value == nullptr) { return 6; }
  auto const setting = std::string_view{value};
  if (setting == "6") { return 6; }
  if (setting == "8") { return 8; }
  throw std::invalid_argument{"LIBCUDF_SEGMENTED_STRING_SORT_BYTES_PER_PASS must be either 6 or 8"};
}

[[nodiscard]] inline segmented_string_sort_config parse_segmented_string_sort_config(
  string_sort_algorithm algorithm,
  char const* bytes_per_pass,
  char const* radix_percent,
  char const* max_radix_passes,
  char const* known_prefix,
  char const* finish_threshold,
  char const* rle_policy,
  char const* rle_min_run_length,
  char const* rle_min_coverage_percent,
  char const* rle_min_equal_percent,
  char const* trace,
  char const* compact_radix = nullptr)
{
  auto config               = segmented_string_sort_config{};
  config.bytes_per_pass     = parse_bytes_per_pass(bytes_per_pass);
  config.radix_percent      = parse_integer_setting(radix_percent, 1, 100, 100);
  config.max_radix_passes   = parse_integer_setting(max_radix_passes, 0, 255, 4);
  config.known_prefix       = parse_integer_setting(known_prefix, 0, 1, 0) != 0;
  config.finish_threshold   = parse_integer_setting(finish_threshold, 2, 1024, 32);
  auto const default_policy = algorithm == string_sort_algorithm::SEGMENTED_RLE ? int{1} : int{0};
  config.rle_policy =
    static_cast<segmented_rle_policy>(parse_integer_setting(rle_policy, 0, 2, default_policy));
  config.rle_min_run_length =
    parse_integer_setting(rle_min_run_length, 2, std::numeric_limits<int>::max(), 32);
  config.rle_min_coverage_percent = parse_integer_setting(rle_min_coverage_percent, 0, 100, 25);
  config.rle_min_equal_percent    = parse_integer_setting(rle_min_equal_percent, 0, 100, 10);
  config.trace                    = parse_integer_setting(trace, 0, 1, 0) != 0;
  config.compact_radix            = parse_integer_setting(compact_radix, 0, 1, 0) != 0;
  return config;
}

/**
 * @brief Parses `LIBCUDF_STRING_SORT_ALGORITHM`.
 *
 * Unset and invalid values select the production prefix implementation.
 */
[[nodiscard]] inline string_sort_algorithm parse_string_sort_algorithm(char const* value)
{
  if (value == nullptr) { return string_sort_algorithm::PREFIX; }
  auto const setting = std::string_view{value};
  if (setting == "1") { return string_sort_algorithm::SEGMENTED; }
  if (setting == "2") { return string_sort_algorithm::SEGMENTED_RLE; }
  return string_sort_algorithm::PREFIX;
}

/**
 * @brief Returns the process-configured experimental string-sort implementation.
 *
 * The environment is read once. Applications must set it before the first libcudf sort call.
 */
[[nodiscard]] inline string_sort_algorithm configured_string_sort_algorithm()
{
  static auto const selected =
    parse_string_sort_algorithm(std::getenv("LIBCUDF_STRING_SORT_ALGORITHM"));
  return selected;
}

[[nodiscard]] inline segmented_string_sort_config const& configured_segmented_string_sort()
{
  static auto const config = parse_segmented_string_sort_config(
    configured_string_sort_algorithm(),
    std::getenv("LIBCUDF_SEGMENTED_STRING_SORT_BYTES_PER_PASS"),
    std::getenv("LIBCUDF_SEGMENTED_STRING_SORT_RADIX_PERCENT"),
    std::getenv("LIBCUDF_SEGMENTED_STRING_SORT_MAX_RADIX_PASSES"),
    std::getenv("LIBCUDF_SEGMENTED_STRING_SORT_KNOWN_PREFIX"),
    std::getenv("LIBCUDF_SEGMENTED_STRING_SORT_FINISH_THRESHOLD"),
    std::getenv("LIBCUDF_SEGMENTED_STRING_SORT_RLE_POLICY"),
    std::getenv("LIBCUDF_SEGMENTED_STRING_SORT_RLE_MIN_RUN_LENGTH"),
    std::getenv("LIBCUDF_SEGMENTED_STRING_SORT_RLE_MIN_COVERAGE_PERCENT"),
    std::getenv("LIBCUDF_SEGMENTED_STRING_SORT_RLE_MIN_EQUAL_PERCENT"),
    std::getenv("LIBCUDF_SEGMENTED_STRING_SORT_TRACE"),
    std::getenv("LIBCUDF_SEGMENTED_STRING_SORT_COMPACT_RADIX"));
  return config;
}

}  // namespace cudf::detail
