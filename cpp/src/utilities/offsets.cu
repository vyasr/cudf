/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/sizes_to_offsets_iterator.cuh>

#include <cuda/iterator>

namespace cudf::detail {

template CUDF_EXPORT std::pair<std::unique_ptr<column>, size_type>
make_offsets_child_column<size_type*>(size_type*,
                                      size_type*,
                                      cuda::stream_ref,
                                      cudf::memory_resources);

template CUDF_EXPORT std::pair<std::unique_ptr<column>, size_type>
make_offsets_child_column<size_type const*>(size_type const*,
                                            size_type const*,
                                            cuda::stream_ref,
                                            cudf::memory_resources);

template CUDF_EXPORT std::pair<std::unique_ptr<column>, size_type>
  make_offsets_child_column<cuda::constant_iterator<size_type>>(cuda::constant_iterator<size_type>,
                                                                cuda::constant_iterator<size_type>,
                                                                cuda::stream_ref,
                                                                cudf::memory_resources);

}  // namespace cudf::detail
