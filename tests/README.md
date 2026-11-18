# Tests

The test suite of fisk, built as one program, `fisk_tests`.

## Building and running

The tests are built by default when fisk is built on its own (`FISK_BUILD_TESTS`). In the main directory of the repository:

```
cmake -B build && cmake --build build -j
ctest --test-dir build
```

or `make`, which builds the tests and the benchmarks in `build/`. The tests can also run directly as `./bin/fisk_tests`, which first prints the CPU and the instruction sets of the build, then runs all tests.

The tests are built with `-march=native`, so that they cover the code paths of all instruction sets of the computer that builds them. The others are not compiled in, and their tests do not exist in that build. CI runs them on Linux, macOS (with Neon), and Windows, with each instruction set of its runners, and the AVX-512 code paths under the Intel Software Development Emulator, see [.github/workflows/test.yaml](../.github/workflows/test.yaml).

## Layout

The test files mostly mirror the library: `tests/<dir>/test_<header>.cpp` tests `include/fisk/<dir>/<header>.hpp`, sometimes split into several files by input check or by span, as each of these files needs a lot of memory to compile. The shared code:

 - `testing.hpp`: a minimal test framework in the style of GoogleTest (`TEST`, `EXPECT_*`, `ASSERT_*`), without dependencies.
 - `oracle.hpp`: the ground truth that the results are checked against, built from the characters with plain per-character rules, independent of the encoders, packers, and extraction functions under test.
 - `corpus.hpp`: random sequences, and the placement of invalid characters in them.
 - `simd_lanes.hpp`: SIMD vectors of raw k-mer words, for the checks of the vectors of k-mers.
 - `test_public_headers.cpp`: that `fisk/fisk.hpp` compiles on its own.

The tests define `FISK_ALWAYS_INLINE_FOR_EACH` as empty, as forcing the many instantiations of the extraction loops inline costs too much compile time. The `extraction/test_forced_inline_*.cpp` files compile each loop once forced inline, as programs that use fisk get them.
