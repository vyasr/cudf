/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/hashing.hpp>
#include <cudf/types.hpp>

namespace cudf::hashing::detail {

extern __device__ hash_value_type murmur_jit_hash_dispatcher(column_device_view col,
                                                             uint32_t seed,
                                                             bool nullable,
                                                             size_type row_index);

template <typename T>
extern __device__ hash_value_type
murmur_jit_hasher(column_device_view col, uint32_t seed, bool nullable, size_type row_index);

}  // namespace cudf::hashing::detail
