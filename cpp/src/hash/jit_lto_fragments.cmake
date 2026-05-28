# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# cmake-lint: disable=C0113,E1120
# =============================================================================
# murmurhash JIT Fragment Registration
# =============================================================================

include(${CMAKE_SOURCE_DIR}/librtcx/compute_matrix_product.cmake)
#
# This file registers murmurhash3_x86_32 JIT fragments using the generic add_fragment() macro from
# librtcx/fragments.cmake.
#
# Total fragments registered for murmurhash3_x86_32: 28 hasher fragments (one per supported type) 28
# noop fragments (one per supported type — for absent types) 2 dispatcher variants (all_types,
# int32_only) 1 entry kernel = 59 total compile units
#
# Build time impact: Each fragment is a separate nvcc -dlto -fatbin invocation. With parallel builds
# (-j N), the effective compile cost is ~1-2 fragments deep. Estimated: 59 fragments × ~5s each =
# ~5-10 minutes on a 32-core machine.
# =============================================================================

# =============================================================================
# Murmurhash LTO fragment registrations
# =============================================================================

set(_MURMURHASH_KERNEL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/hash/jit_lto_kernels")
set(_MURMURHASH_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated_jit_lto/murmurhash")

compute_matrix_product(
  _MURMURHASH_HASHER_MATRIX MATRIX_JSON_FILE
  "${_MURMURHASH_KERNEL_SOURCE_DIR}/murmurhash_hasher_matrix.json"
)
string(JSON _MURMURHASH_HASHER_MATRIX_LEN LENGTH "${_MURMURHASH_HASHER_MATRIX}")
math(EXPR _MURMURHASH_HASHER_MATRIX_LAST "${_MURMURHASH_HASHER_MATRIX_LEN} - 1")

# Hasher and noop fragments
foreach(_MURMURHASH_INDEX RANGE "${_MURMURHASH_HASHER_MATRIX_LAST}")
  string(JSON _MURMURHASH_MATRIX_ENTRY GET "${_MURMURHASH_HASHER_MATRIX}" "${_MURMURHASH_INDEX}")
  populate_matrix_variables("${_MURMURHASH_MATRIX_ENTRY}")

  set(_MURMURHASH_HASHER_SOURCE
      "${_MURMURHASH_GENERATED_DIR}/murmurhash_hasher_${abbrev}_fragment.cu"
  )
  configure_file(
    "${_MURMURHASH_KERNEL_SOURCE_DIR}/murmurhash_hasher_fragment.cu.in"
    "${_MURMURHASH_HASHER_SOURCE}" @ONLY
  )
  add_fragment(
    cudf_hash_fragments
    FRAGMENT
    murmurhash_hasher_${abbrev}
    SOURCE
    ${_MURMURHASH_HASHER_SOURCE}
    LINK_LIBRARIES
    CCCL::CCCL
    rapids_logger::rapids_logger
    rmm::rmm
    $<BUILD_LOCAL_INTERFACE:BS::thread_pool>
    $<BUILD_LOCAL_INTERFACE:nvtx3::nvtx3-cpp>
    $<BUILD_LOCAL_INTERFACE:cuco::cuco>
    ZLIB::ZLIB
    nvcomp::nvcomp
    kvikio::kvikio
    nanoarrow::nanoarrow
    zstd
    INCLUDE_DIRECTORIES
    "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/src>"
    COMPILE_DEFINITIONS
    CUDF_DISABLE_EXPORTS
    ARRAY_IDS
    fragment_abbrev
    ARRAY_VALUES
    ${abbrev}
  )

  set(_MURMURHASH_NOOP_SOURCE "${_MURMURHASH_GENERATED_DIR}/murmurhash_noop_${abbrev}_fragment.cu")
  configure_file(
    "${_MURMURHASH_KERNEL_SOURCE_DIR}/murmurhash_hasher_noop_fragment.cu.in"
    "${_MURMURHASH_NOOP_SOURCE}" @ONLY
  )
  add_fragment(
    cudf_hash_fragments
    FRAGMENT
    murmurhash_noop_${abbrev}
    SOURCE
    ${_MURMURHASH_NOOP_SOURCE}
    LINK_LIBRARIES
    CCCL::CCCL
    rapids_logger::rapids_logger
    rmm::rmm
    $<BUILD_LOCAL_INTERFACE:BS::thread_pool>
    $<BUILD_LOCAL_INTERFACE:nvtx3::nvtx3-cpp>
    $<BUILD_LOCAL_INTERFACE:cuco::cuco>
    ZLIB::ZLIB
    nvcomp::nvcomp
    kvikio::kvikio
    nanoarrow::nanoarrow
    zstd
    INCLUDE_DIRECTORIES
    "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/src>"
    COMPILE_DEFINITIONS
    CUDF_DISABLE_EXPORTS
    ARRAY_IDS
    fragment_abbrev
    ARRAY_VALUES
    ${abbrev}
  )
endforeach()

# Dispatcher fragments (2)
compute_matrix_product(
  _MURMURHASH_DISPATCH_MATRIX MATRIX_JSON_FILE
  "${_MURMURHASH_KERNEL_SOURCE_DIR}/murmurhash_dispatch_matrix.json"
)
string(JSON _MURMURHASH_DISPATCH_MATRIX_LEN LENGTH "${_MURMURHASH_DISPATCH_MATRIX}")
math(EXPR _MURMURHASH_DISPATCH_MATRIX_LAST "${_MURMURHASH_DISPATCH_MATRIX_LEN} - 1")

foreach(_MURMURHASH_INDEX RANGE "${_MURMURHASH_DISPATCH_MATRIX_LAST}")
  string(JSON _MURMURHASH_MATRIX_ENTRY GET "${_MURMURHASH_DISPATCH_MATRIX}" "${_MURMURHASH_INDEX}")
  populate_matrix_variables("${_MURMURHASH_MATRIX_ENTRY}")

  set(_MURMURHASH_DISPATCH_SOURCE
      "${_MURMURHASH_GENERATED_DIR}/murmurhash_dispatch_${suffix}_fragment.cu"
  )
  configure_file(
    "${_MURMURHASH_KERNEL_SOURCE_DIR}/murmurhash_dispatch_fragment.cu.in"
    "${_MURMURHASH_DISPATCH_SOURCE}" @ONLY
  )
  add_fragment(
    cudf_hash_fragments
    FRAGMENT
    murmurhash_dispatch_${suffix}
    SOURCE
    ${_MURMURHASH_DISPATCH_SOURCE}
    LINK_LIBRARIES
    CCCL::CCCL
    rapids_logger::rapids_logger
    rmm::rmm
    $<BUILD_LOCAL_INTERFACE:BS::thread_pool>
    $<BUILD_LOCAL_INTERFACE:nvtx3::nvtx3-cpp>
    $<BUILD_LOCAL_INTERFACE:cuco::cuco>
    ZLIB::ZLIB
    nvcomp::nvcomp
    kvikio::kvikio
    nanoarrow::nanoarrow
    zstd
    INCLUDE_DIRECTORIES
    "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/src>"
    COMPILE_DEFINITIONS
    CUDF_DISABLE_EXPORTS
    ARRAY_IDS
    fragment_variant
    ARRAY_VALUES
    ${suffix}
  )
endforeach()

# Entry kernel (1)
compute_matrix_product(
  _MURMURHASH_ENTRY_MATRIX MATRIX_JSON_FILE
  "${_MURMURHASH_KERNEL_SOURCE_DIR}/murmurhash_entry_matrix.json"
)
string(JSON _MURMURHASH_ENTRY_MATRIX_LEN LENGTH "${_MURMURHASH_ENTRY_MATRIX}")
math(EXPR _MURMURHASH_ENTRY_MATRIX_LAST "${_MURMURHASH_ENTRY_MATRIX_LEN} - 1")

foreach(_MURMURHASH_INDEX RANGE "${_MURMURHASH_ENTRY_MATRIX_LAST}")
  string(JSON _MURMURHASH_MATRIX_ENTRY GET "${_MURMURHASH_ENTRY_MATRIX}" "${_MURMURHASH_INDEX}")
  populate_matrix_variables("${_MURMURHASH_MATRIX_ENTRY}")

  set(_MURMURHASH_ENTRY_SOURCE "${_MURMURHASH_GENERATED_DIR}/murmurhash_entry_kernel.cu")
  configure_file(
    "${_MURMURHASH_KERNEL_SOURCE_DIR}/murmurhash_entry_kernel.cu.in" "${_MURMURHASH_ENTRY_SOURCE}"
    @ONLY
  )

  add_fragment(
    cudf_hash_fragments
    FRAGMENT
    murmurhash_entry
    SOURCE
    ${_MURMURHASH_ENTRY_SOURCE}
    KERNEL_ONLY
    ENTRY_NAME
    cudf_murmurhash3_x86_32_jit_link_kernel
    LINK_LIBRARIES
    CCCL::CCCL
    rapids_logger::rapids_logger
    rmm::rmm
    $<BUILD_LOCAL_INTERFACE:BS::thread_pool>
    $<BUILD_LOCAL_INTERFACE:nvtx3::nvtx3-cpp>
    $<BUILD_LOCAL_INTERFACE:cuco::cuco>
    ZLIB::ZLIB
    nvcomp::nvcomp
    kvikio::kvikio
    nanoarrow::nanoarrow
    zstd
    INCLUDE_DIRECTORIES
    "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/src>"
    COMPILE_DEFINITIONS
    CUDF_DISABLE_EXPORTS
    ARRAY_IDS
    fragment_role
    ARRAY_VALUES
    ${k}
  )
endforeach()

# Composite fragments must also be generated per architecture to avoid nvJitLink's multi-arch LTO
# fatbin limitation. The runtime planner uses individual per-architecture fragments until those
# composites are added.
