# libcudf binary-size measurements

All measurements use the release CUDA 12.9 build configuration and its complete architecture list.
The device-code report is produced by `scripts/report_libcudf_binary_size.sh` when `cuobjdump` is
available in the devcontainer.

| Commit | Change | `libcudf.so` bytes | `.nv_fatbin` bytes | Delta |
| --- | --- | ---: | ---: | ---: |
| `upstream/main` | Baseline | 1,060,033,664 | 878,906,256 | — |
| `refactor(hash-join): centralize HashCSR fixed-signature kernels` | One owner TU for build-fill and retrieve kernels | 1,059,792,560 | 878,659,976 | -241,104 library bytes; -246,280 fatbin bytes |
| `refactor(hash-join): centralize HashCSR count instantiations` | One owner TU for join-size probe-count instantiations | 1,057,650,528 | 876,548,720 | -2,142,032 library bytes; -2,111,256 fatbin bytes |

## Validation notes

- The HashCSR owner TU and all affected caller TUs compiled successfully with a capped `-j16`
  distributed-sccache build.
- The join-size owner TU and its three call-site TUs compiled successfully with a capped `-j16`
  build. Its final link was cache-only: 10 requests, no compilations or failures.
- `JOIN_TEST` and `STREAM_JOIN_TEST` require GPU CTest resource slots. The current devcontainer
  advertises zero slots, so these tests were not runnable in this environment.
