# Copyright (c) 2023-2024, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.26.4)

set(CMAKE_CUDA_ARCHITECTURES native)
set(cudf_ROOT /home/coder/cudf/cpp/build/latest)
project(
  nested_types
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

# Configure your project here
add_executable(stream_example streams.cu)
target_link_libraries(stream_example PRIVATE cudf::cudf)
# target_compile_features(deduplication PRIVATE cxx_std_17)
list(APPEND CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
target_compile_options(stream_example PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>")
