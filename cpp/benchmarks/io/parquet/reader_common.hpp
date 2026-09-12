/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <benchmarks/io/cuio_common.hpp>

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

constexpr cudf::size_type num_cols = 64;

/**
 * @brief Translate a `null_percent` axis value into a `data_profile` null probability.
 *
 * The Parquet level prepass only does work when a page actually carries definition
 * levels, so null frequency decides whether the feature is engaged at all. The axis
 * distinguishes two cases that are easy to conflate:
 *
 * - `-1` produces no validity mask, so the column is written `required` and no level
 *   state is allocated. This is the prepass overhead floor.
 * - `0` produces an all-valid validity mask, so the column is written `optional` and
 *   the prepass runs a full scan to publish an identity map. This is the pure-overhead
 *   worst case.
 */
[[nodiscard]] std::optional<double> null_probability_from_percent(int64_t null_percent);

/**
 * @brief Resolve a `prepass_mode` axis value to a `LIBCUDF_PARQUET_LEVEL_PREPASS` value.
 *
 * Returns `std::nullopt` for "default", meaning the ambient environment is left alone.
 * A value beginning with "0x" is passed through verbatim so one-off bitmasks can be
 * requested from the command line without recompiling.
 */
[[nodiscard]] std::optional<std::string> prepass_selector_for_mode(std::string_view mode);

// Every benchmark that calls `parquet_read_common` must register a "prepass_mode" string
// axis; those whose body honours it also register a "null_percent" int64 axis. Both carry a
// single default value so registering them does not multiply the existing benchmark matrix.

void parquet_read_common(cudf::size_type num_rows_to_read,
                         cudf::size_type num_cols_to_read,
                         cuio_source_sink_pair& source_sink,
                         nvbench::state& state);
