/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdlib>
#include <string_view>

namespace cudf::detail {

/**
 * @brief Experimental single-column string sorting implementations.
 */
enum class string_sort_algorithm { PREFIX, SEGMENTED, SEGMENTED_RLE };

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

}  // namespace cudf::detail
