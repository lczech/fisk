# ==================================================================================================
#     Basic Setup
# ==================================================================================================

cmake_minimum_required(VERSION 3.20)
project(fisk LANGUAGES CXX)

# Use C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Put final binaries in bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Ensure a default build type if none is given
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Whether fisk is the top-level project (as opposed to being pulled in via
# add_subdirectory()/FetchContent() by another project). Equivalent to CMake
# 3.21's PROJECT_IS_TOP_LEVEL, spelled out here to keep the 3.20 minimum.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(FISK_IS_TOP_LEVEL_DEFAULT ON)
else()
    set(FISK_IS_TOP_LEVEL_DEFAULT OFF)
endif()

# ==================================================================================================
#     Library
# ==================================================================================================

# Header-only library. Consumers link against fisk::fisk and get include/ on their path;
# nothing here is compiled, and nothing here should force compiler flags onto consumers.
add_library(fisk INTERFACE)
add_library(fisk::fisk ALIAS fisk)

target_include_directories(fisk INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(fisk INTERFACE cxx_std_20)

# ==================================================================================================
#     Detect Instruction Sets
# ==================================================================================================

# Each instruction set below follows the identical shape: a compiler-capability check, an
# opt-in FISK_ENABLE_* option, and, if enabled and supported, a compile flag plus a FISK_HAS_*
# define. The best way to get "everything this build machine actually has" is -march=native, which
# only ever enables extensions the CPU building reports. Those predefine the compiler flags such as
#  __AVX2__/__BMI2__/etc., which core/intrinsics.hpp checks.
#
# MSVC has no per-extension flags the way GCC/Clang do -- only /arch: tiers that bundle several
# extensions together, so the checks and the flags applied below differ for it:
#   - SSE2 is the x64 baseline (always present), so it needs no flag and no check.
#   - PCLMUL has no gating flag on MSVC at all; the intrinsic header is available unconditionally,
#     so it is treated as always "supported" here and relies purely on FISK_ENABLE_PCLMUL's own
#     opt-in warning, same trust model the option already documents for every compiler.
#   - BMI2 has no standalone tier either -- only /arch:AVX2 enables it. So FISK_ENABLE_BMI2 alone
#     (without FISK_ENABLE_AVX2) still applies /arch:AVX2, but only defines FISK_HAS_BMI2: the
#     library's AVX2 code paths stay off unless FISK_ENABLE_AVX2 was separately requested.
#   - AVX512F and AVX512BW are not independently selectable either; both come from /arch:AVX512.
include(CheckCXXCompilerFlag)

# -march=native is used below (fisk_tests, fisk_benchmarks) to activate whatever ISA extensions
# the build machine actually has. It is not universally reliable, though: some older Clang
# versions don't support it on AArch64 at all, some older GCC versions on Apple Silicon resolve
# it to a CPU name their own backend doesn't recognize (e.g. "apple-m1"), and on some very new
# CPU/compiler combinations it can expand to a feature combination the compiler itself rejects
# under -Werror (e.g. Clang's AVX10 promotion warning). Checked once here and applied only where
# it actually works, falling back to no flag (i.e. whatever ISA the compiler defaults to) rather
# than failing the whole build on hosts where it doesn't.
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)

# The AVX10 case above needs its own separate handling: the trivial program the check above
# compiles doesn't reach whatever code path trips the "invalid feature combination" diagnostic,
# so COMPILER_SUPPORTS_MARCH_NATIVE alone doesn't catch it; it only shows up compiling the real
# sources. Silence just this one diagnostic (rather than weakening -Werror overall) where the
# compiler recognizes it at all.
check_cxx_compiler_flag(
    "-Wno-error=invalid-feature-combination" COMPILER_SUPPORTS_NO_ERROR_INVALID_FEATURE_COMBINATION
)

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    check_cxx_compiler_flag("/arch:AVX2"   COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("/arch:AVX512" COMPILER_SUPPORTS_AVX512)
    set(COMPILER_SUPPORTS_BMI2 ${COMPILER_SUPPORTS_AVX2})
    set(COMPILER_SUPPORTS_SSE2 TRUE)
    set(COMPILER_SUPPORTS_PCLMUL TRUE)
    set(COMPILER_SUPPORTS_AVX512BW ${COMPILER_SUPPORTS_AVX512})
else()
    check_cxx_compiler_flag("-mbmi2"     COMPILER_SUPPORTS_BMI2)
    check_cxx_compiler_flag("-mpclmul"   COMPILER_SUPPORTS_PCLMUL)
    check_cxx_compiler_flag("-msse2"     COMPILER_SUPPORTS_SSE2)
    check_cxx_compiler_flag("-mavx2"     COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-mavx512f"  COMPILER_SUPPORTS_AVX512)
    check_cxx_compiler_flag("-mavx512bw" COMPILER_SUPPORTS_AVX512BW)
endif()

option(FISK_ENABLE_BMI2   "Force-enable fisk's BMI2-accelerated code paths (PEXT); only safe if the deployment target actually has BMI2" OFF)
option(FISK_ENABLE_PCLMUL "Force-enable fisk's PCLMUL-accelerated code paths; only safe if the deployment target actually has PCLMUL" OFF)
option(FISK_ENABLE_SSE2   "Force-enable fisk's SSE2-accelerated code paths; only safe if the deployment target actually has SSE2" OFF)
option(FISK_ENABLE_AVX2   "Force-enable fisk's AVX2-accelerated code paths; only safe if the deployment target actually has AVX2" OFF)
option(FISK_ENABLE_AVX512 "Force-enable fisk's AVX512-accelerated code paths; only safe if the deployment target actually has AVX512" OFF)
option(FISK_ENABLE_NEON   "Force-enable fisk's NEON-accelerated code paths (ARM64); only safe if the deployment target actually has NEON" OFF)

if (FISK_ENABLE_BMI2 AND COMPILER_SUPPORTS_BMI2)
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(fisk INTERFACE "/arch:AVX2")
    else()
        target_compile_options(fisk INTERFACE "-mbmi2")
    endif()
    target_compile_definitions(fisk INTERFACE FISK_HAS_BMI2=1)
endif()

if (FISK_ENABLE_PCLMUL AND COMPILER_SUPPORTS_PCLMUL)
    if (NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(fisk INTERFACE "-mpclmul")
    endif()
    target_compile_definitions(fisk INTERFACE FISK_HAS_CLMUL=1)
endif()

if (FISK_ENABLE_SSE2 AND COMPILER_SUPPORTS_SSE2)
    if (NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(fisk INTERFACE "-msse2")
    endif()
    target_compile_definitions(fisk INTERFACE FISK_HAS_SSE2=1)
endif()

if (FISK_ENABLE_AVX2 AND COMPILER_SUPPORTS_AVX2)
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(fisk INTERFACE "/arch:AVX2")
    else()
        target_compile_options(fisk INTERFACE "-mavx2")
    endif()
    target_compile_definitions(fisk INTERFACE FISK_HAS_AVX2=1)
endif()

# AVX512BW is required alongside AVX512F: seq_pack's Msb byte-reversal uses _mm512_shuffle_epi8,
# which needs BW. It's present on essentially all mainstream AVX512 hardware since Skylake-X/Ice
# Lake -- only the earliest, now-obsolete Xeon Phi "Knights Landing" parts are F-only -- so this is
# folded into the single FISK_ENABLE_AVX512 option rather than a separate opt-in tier.
if (FISK_ENABLE_AVX512 AND COMPILER_SUPPORTS_AVX512 AND COMPILER_SUPPORTS_AVX512BW)
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(fisk INTERFACE "/arch:AVX512")
    else()
        target_compile_options(fisk INTERFACE "-mavx512f" "-mavx512bw")
    endif()
    target_compile_definitions(fisk INTERFACE FISK_HAS_AVX512=1)
endif()

# NEON is not an optional x86 extension selectable via a compile flag the way the others are —
# it is a different base architecture. There is nothing to check_cxx_compiler_flag() here; we
# just define the macro when actually building for ARM64, guarded so this never fires on x86.
if (FISK_ENABLE_NEON AND (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64"))
    target_compile_definitions(fisk INTERFACE FISK_HAS_NEON=1)
endif()

# ==================================================================================================
#     Benchmarks
# ==================================================================================================

# Only built by default when fisk is the top-level project (i.e. not when pulled in as a
# dependency by another project), but can be forced either way via -DFISK_BUILD_BENCHMARKS=.
option(FISK_BUILD_BENCHMARKS "Build the fisk benchmark suite" ${FISK_IS_TOP_LEVEL_DEFAULT})

if(FISK_BUILD_BENCHMARKS)
    add_executable(fisk_benchmarks benchmarks/main.cpp)
    target_include_directories(fisk_benchmarks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
    target_link_libraries(fisk_benchmarks PRIVATE fisk::fisk)

    # Optimization and warning flags are private to the benchmark binary: they are build-quality
    # choices for this repo, not correctness requirements for the library, and must not propagate
    # to anything that links fisk::fisk.
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(fisk_benchmarks PRIVATE
            $<$<CONFIG:Release>:-O3>
        )
        if (COMPILER_SUPPORTS_MARCH_NATIVE)
            target_compile_options(fisk_benchmarks PRIVATE
                $<$<CONFIG:Release>:-march=native>
            )
        endif()

        target_compile_options(fisk_benchmarks PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wshadow
            -Wconversion
            -Wsign-conversion
            -Wnull-dereference
            -Wdouble-promotion
            -Werror
        )
        if (COMPILER_SUPPORTS_NO_ERROR_INVALID_FEATURE_COMBINATION)
            target_compile_options(fisk_benchmarks PRIVATE
                -Wno-error=invalid-feature-combination
            )
        endif()
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(fisk_benchmarks PRIVATE
            /W4
            /WX
            # C4714: function marked as [[msvc::forceinline]]/__forceinline not inlined. Some of
            # the for_each_kmer_*() loops (see FISK_ALWAYS_INLINE_FOR_EACH, core/intrinsics.hpp)
            # are already known to sometimes decline this on GCC/Clang too (see the comment on
            # for_each_kmer_packed_aligned_narrow_impl_() in kmer_extract/packed.hpp); MSVC
            # declining should stay a silent no-op rather than fail the /WX build.
            /wd4714
            # C4324: struct padded due to an alignas specifier. Expected and harmless for the
            # SIMD kernel structs (bit_extract/simd.hpp), which need natural alignment for their
            # vector register members. GCC/Clang don't warn on this at all.
            /wd4324
        )
    endif()
endif()

# ==================================================================================================
#     Tests
# ==================================================================================================

# Only built by default when fisk is the top-level project, same rationale as
# FISK_BUILD_BENCHMARKS above, and can likewise be forced either way via -DFISK_BUILD_TESTS=.
option(FISK_BUILD_TESTS "Build the fisk test suite" ${FISK_IS_TOP_LEVEL_DEFAULT})

if(FISK_BUILD_TESTS)
    enable_testing()

    add_executable(fisk_tests
        tests/main.cpp
        tests/test_core.cpp
        tests/test_char_encoder.cpp
        tests/test_bit_extract.cpp
        tests/test_seq_pack.cpp
        tests/test_kmer.cpp
        tests/test_kmer_extract.cpp
        tests/test_kmer_extract_packed.cpp
        tests/test_kmer_extract_packed_simd.cpp
    )
    target_include_directories(fisk_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(fisk_tests PRIVATE fisk::fisk)

    # Same rationale as fisk_benchmarks above: build-quality choices private to this binary.
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        # Activate whatever ISA extensions this build machine actually has, for the test
        # suite only, so that the library itself stays portable. Skipped where unsupported
        # (see COMPILER_SUPPORTS_MARCH_NATIVE above) rather than failing the build.
        if (COMPILER_SUPPORTS_MARCH_NATIVE)
            target_compile_options(fisk_tests PRIVATE
                -march=native
            )
        endif()

        # Keep assert() live regardless of build type. CMAKE_BUILD_TYPE defaults to Release, which
        # carries -DNDEBUG, so without this the library's preconditions -- including the k-mer
        # convention checks in kmer_cast() -- would be compiled out of the very suite meant to
        # exercise them. target_compile_options land last on the command line, so this wins.
        target_compile_options(fisk_tests PRIVATE
            -UNDEBUG
        )

        target_compile_options(fisk_tests PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wshadow
            -Wconversion
            -Wsign-conversion
            -Wnull-dereference
            -Wdouble-promotion
            -Werror
        )
        if (COMPILER_SUPPORTS_NO_ERROR_INVALID_FEATURE_COMBINATION)
            target_compile_options(fisk_tests PRIVATE
                -Wno-error=invalid-feature-combination
            )
        endif()
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        # Same rationale as the GCC/Clang -UNDEBUG above: keep assert() live in the default
        # Release build type, so the test suite still exercises the library's preconditions.
        target_compile_options(fisk_tests PRIVATE
            /UNDEBUG
            /W4
            /WX
            # See the matching /wd4714 and /wd4324 on fisk_benchmarks above for why.
            /wd4714
            /wd4324
        )
    endif()

    add_test(NAME fisk_tests COMMAND fisk_tests)
endif()
