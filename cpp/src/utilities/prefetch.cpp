/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/prefetch.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/prefetch.hpp>

#include <iostream>

namespace cudf::experimental::prefetch {

namespace detail {

PrefetchConfig& PrefetchConfig::instance()
{
  static PrefetchConfig instance;
  return instance;
}

bool PrefetchConfig::get(std::string_view key)
{
  // Default to not prefetching
  if (config_values.find(key.data()) == config_values.end()) {
    return (config_values[key.data()] = false);
  }
  return config_values[key.data()];
}
void PrefetchConfig::set(std::string_view key, bool value) { config_values[key.data()] = value; }

void prefetch(std::string_view key, void const* ptr, std::size_t size)
{
  if (PrefetchConfig::instance().get(key)) {
    if (PrefetchConfig::instance().debug) {
      std::cerr << "Prefetching " << size << " bytes for key " << key << " at location " << ptr
                << std::endl;
    }

    rmm::prefetch(ptr, size, rmm::get_current_cuda_device(), cudf::get_default_stream());
  }
}

}  // namespace detail

void enable_prefetching(std::string_view key) { detail::PrefetchConfig::instance().set(key, true); }

void disable_prefetching(std::string_view key)
{
  detail::PrefetchConfig::instance().set(key, false);
}

void prefetch_debugging(bool enable) { detail::PrefetchConfig::instance().debug = enable; }
}  // namespace cudf::experimental::prefetch
