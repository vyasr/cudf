// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cudf/table/table_view.hpp>
#include <cudf/utilities/export.hpp>

#include <rtcx.hpp>

#include <vector>

namespace cudf::hashing::detail {

CUDF_EXPORT std::vector<rtcx::memory_fragment> plan_murmurhash_fragments(
  cudf::table_view const& input);

}  // namespace cudf::hashing::detail
