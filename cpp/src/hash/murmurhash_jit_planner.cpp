// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0
// Fragment planner for murmurhash3_x86_32 JIT-linked kernel.
//
// the table's column types. The fragments are then linked by librtcx via
// nvJitLink. First-link latency is typically 5-50ms depending on GPU and
// fragment count; subsequent calls with the same schema hit the cache
// (both in-process and on-disk) at near-zero cost.

#include "murmurhash_jit_planner.hpp"

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda_runtime_api.h>

#include <cudf_hash_fragments.hpp>

#include <array>
#include <cstdint>
#include <format>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>

namespace cudf::hashing::detail {
namespace {

using type_id = cudf::type_id;

auto embedded_fragment_files() -> std::span<std::uint8_t const>
{
  static auto const files = rtcx::decompress_blob(cudf_hash_fragments::files,
                                                  cudf_hash_fragments::files_uncompressed_size,
                                                  cudf_hash_fragments::files_compression);
  return {files.data(), files.size()};
}

auto make_fragment(std::size_t idx, char const*) -> rtcx::memory_fragment
{
  auto const& range = cudf_hash_fragments::file_ranges[idx];
  return rtcx::memory_fragment{embedded_fragment_files().subspan(range[0], range[1]),
                               rtcx::binary_type::FATBIN,
                               cudf_hash_fragments::file_ids[idx]};
}

auto find_fragment_index(std::string_view name) -> std::optional<std::size_t>
{
  constexpr auto num_fragments =
    sizeof(cudf_hash_fragments::file_ids) / sizeof(cudf_hash_fragments::file_ids[0]);
  for (std::size_t idx = 0; idx < num_fragments; ++idx) {
    if (cudf_hash_fragments::file_ids[idx] == name) { return idx; }
  }
  return std::nullopt;
}

bool fragment_arch_available(std::string_view arch)
{
  constexpr auto num_fragments =
    sizeof(cudf_hash_fragments::fragment_arch) / sizeof(cudf_hash_fragments::fragment_arch[0]);
  for (std::size_t idx = 0; idx < num_fragments; ++idx) {
    if (cudf_hash_fragments::fragment_arch[idx] == arch) { return true; }
  }
  return false;
}

auto current_device_fragment_suffix() -> std::string
{
  int device   = 0;
  int cc_major = 0;
  int cc_minor = 0;

  auto check_cuda = [](cudaError_t status) {
    if (status != cudaSuccess) { cudf::detail::throw_cuda_error(status, __FILE__, __LINE__); }
  };

  check_cuda(cudaGetDevice(&device));
  check_cuda(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device));
  check_cuda(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, device));

  auto const base_arch = std::format("sm{}{}", cc_major, cc_minor);
  if (fragment_arch_available(base_arch)) { return std::format("_{}", base_arch); }

  if (cc_major == 9 && cc_minor == 0 && fragment_arch_available("sm90a")) { return "_sm90a"; }
  if (cc_major == 10 && cc_minor == 0 && fragment_arch_available("sm100f")) { return "_sm100f"; }
  if (cc_major == 12 && cc_minor == 0 && fragment_arch_available("sm120a")) { return "_sm120a"; }

  return std::format("_{}", base_arch);
}

auto arch_fragment_index(std::string_view base_name, std::string_view arch_suffix)
  -> std::optional<std::size_t>
{
  auto arch_name = std::string{base_name};
  arch_name += arch_suffix;
  return find_fragment_index(arch_name);
}

struct fragment_names {
  char const* hasher_name;
  char const* noop_name;
};

using fragment_pair = std::pair<type_id, fragment_names>;

constexpr std::array<fragment_pair, 28> supported_type_fragments = {{
  {type_id::INT8, {"murmurhash_hasher_i8", "murmurhash_noop_i8"}},
  {type_id::INT16, {"murmurhash_hasher_i16", "murmurhash_noop_i16"}},
  {type_id::INT32, {"murmurhash_hasher_i32", "murmurhash_noop_i32"}},
  {type_id::INT64, {"murmurhash_hasher_i64", "murmurhash_noop_i64"}},
  {type_id::UINT8, {"murmurhash_hasher_u8", "murmurhash_noop_u8"}},
  {type_id::UINT16, {"murmurhash_hasher_u16", "murmurhash_noop_u16"}},
  {type_id::UINT32, {"murmurhash_hasher_u32", "murmurhash_noop_u32"}},
  {type_id::UINT64, {"murmurhash_hasher_u64", "murmurhash_noop_u64"}},
  {type_id::FLOAT32, {"murmurhash_hasher_f32", "murmurhash_noop_f32"}},
  {type_id::FLOAT64, {"murmurhash_hasher_f64", "murmurhash_noop_f64"}},
  {type_id::BOOL8, {"murmurhash_hasher_b8", "murmurhash_noop_b8"}},
  {type_id::TIMESTAMP_DAYS, {"murmurhash_hasher_ts_day", "murmurhash_noop_ts_day"}},
  {type_id::TIMESTAMP_SECONDS, {"murmurhash_hasher_ts_s", "murmurhash_noop_ts_s"}},
  {type_id::TIMESTAMP_MILLISECONDS, {"murmurhash_hasher_ts_ms", "murmurhash_noop_ts_ms"}},
  {type_id::TIMESTAMP_MICROSECONDS, {"murmurhash_hasher_ts_us", "murmurhash_noop_ts_us"}},
  {type_id::TIMESTAMP_NANOSECONDS, {"murmurhash_hasher_ts_ns", "murmurhash_noop_ts_ns"}},
  {type_id::DURATION_DAYS, {"murmurhash_hasher_du_day", "murmurhash_noop_du_day"}},
  {type_id::DURATION_SECONDS, {"murmurhash_hasher_du_s", "murmurhash_noop_du_s"}},
  {type_id::DURATION_MILLISECONDS, {"murmurhash_hasher_du_ms", "murmurhash_noop_du_ms"}},
  {type_id::DURATION_MICROSECONDS, {"murmurhash_hasher_du_us", "murmurhash_noop_du_us"}},
  {type_id::DURATION_NANOSECONDS, {"murmurhash_hasher_du_ns", "murmurhash_noop_du_ns"}},
  {type_id::DICTIONARY32, {"murmurhash_hasher_dict", "murmurhash_noop_dict"}},
  {type_id::STRING, {"murmurhash_hasher_str", "murmurhash_noop_str"}},
  {type_id::LIST, {"murmurhash_hasher_list", "murmurhash_noop_list"}},
  {type_id::DECIMAL32, {"murmurhash_hasher_dec32", "murmurhash_noop_dec32"}},
  {type_id::DECIMAL64, {"murmurhash_hasher_dec64", "murmurhash_noop_dec64"}},
  {type_id::DECIMAL128, {"murmurhash_hasher_dec128", "murmurhash_noop_dec128"}},
  {type_id::STRUCT, {"murmurhash_hasher_struct", "murmurhash_noop_struct"}},
}};

void collect_nested_logical_types(std::unordered_set<type_id>& ids, column_view const& col)
{
  ids.insert(col.type().id());
  switch (col.type().id()) {
    case type_id::STRUCT:
      for (size_type i = 0; i < col.num_children(); ++i) {
        collect_nested_logical_types(ids, col.child(i));
      }
      break;
    case type_id::LIST: {
      lists_column_view const lcv(col);
      collect_nested_logical_types(ids, lcv.child());
      break;
    }
    case type_id::DICTIONARY32: {
      dictionary_column_view const dcv(col);
      collect_nested_logical_types(ids, dcv.keys());
      break;
    }
    default: break;
  }
}

bool use_int32_only_dispatch(std::unordered_set<type_id> const& present_types)
{
  return present_types.size() == 1u && present_types.count(type_id::INT32) == 1u;
}

bool is_supported_type(type_id id)
{
  for (auto const& [supported_id, _] : supported_type_fragments) {
    if (id == supported_id) { return true; }
  }
  return false;
}

bool has_nested_dictionary(column_view const& col, bool nested = false)
{
  switch (col.type().id()) {
    case type_id::DICTIONARY32: return nested;
    case type_id::STRUCT:
      for (size_type i = 0; i < col.num_children(); ++i) {
        if (has_nested_dictionary(col.child(i), true)) { return true; }
      }
      return false;
    case type_id::LIST: {
      lists_column_view const lcv(col);
      return has_nested_dictionary(lcv.child(), true);
    }
    default: return false;
  }
}

}  // namespace

std::vector<rtcx::memory_fragment> plan_murmurhash_fragments(cudf::table_view const& input)
{
  std::unordered_set<type_id> present_types;
  present_types.reserve(input.num_columns());
  for (auto const& col : input) {
    if (has_nested_dictionary(col)) { return {}; }
    collect_nested_logical_types(present_types, col);
  }
  for (auto const id : present_types) {
    if (!is_supported_type(id)) { return {}; }
  }

  std::vector<rtcx::memory_fragment> fragments;
  auto const arch_suffix = current_device_fragment_suffix();
  bool const int32_only  = use_int32_only_dispatch(present_types);

  fragments.reserve(supported_type_fragments.size() + 2);
  for (auto const& [id, names] : supported_type_fragments) {
    if (int32_only && id != type_id::INT32) { continue; }
    auto const* fragment_name = present_types.count(id) != 0 ? names.hasher_name : names.noop_name;
    auto const fragment_index = arch_fragment_index(fragment_name, arch_suffix);
    if (!fragment_index.has_value()) { return {}; }

    fragments.push_back(make_fragment(*fragment_index, fragment_name));
  }

  auto const dispatch_index = arch_fragment_index(
    int32_only ? "murmurhash_dispatch_int32" : "murmurhash_dispatch_all", arch_suffix);
  auto const entry_index = arch_fragment_index("murmurhash_entry", arch_suffix);
  if (!dispatch_index.has_value() || !entry_index.has_value()) { return {}; }

  fragments.push_back(make_fragment(
    *dispatch_index, int32_only ? "murmurhash_dispatch_int32" : "murmurhash_dispatch_all"));
  fragments.push_back(make_fragment(*entry_index, "murmurhash_entry"));

  return fragments;
}

}  // namespace cudf::hashing::detail
