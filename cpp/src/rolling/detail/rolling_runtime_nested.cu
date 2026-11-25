/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rolling_runtime.cuh"

CUDF_INSTANTIATE_RUNTIME_ROLLING(cudf::string_view)
CUDF_INSTANTIATE_RUNTIME_ROLLING(cudf::list_view)
CUDF_INSTANTIATE_RUNTIME_ROLLING(cudf::struct_view)

#undef CUDF_INSTANTIATE_RUNTIME_ROLLING
