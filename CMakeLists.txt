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
include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-mbmi2"    COMPILER_SUPPORTS_BMI2)
check_cxx_compiler_flag("-mpclmul"  COMPILER_SUPPORTS_PCLMUL)
check_cxx_compiler_flag("-msse2"    COMPILER_SUPPORTS_SSE2)
check_cxx_compiler_flag("-mavx2"    COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mavx512f"  COMPILER_SUPPORTS_AVX512)
check_cxx_compiler_flag("-mavx512bw" COMPILER_SUPPORTS_AVX512BW)

option(FISK_ENABLE_BMI2   "Force-enable fisk's BMI2-accelerated code paths (PEXT); only safe if the deployment target actually has BMI2" OFF)
option(FISK_ENABLE_PCLMUL "Force-enable fisk's PCLMUL-accelerated code paths; only safe if the deployment target actually has PCLMUL" OFF)
option(FISK_ENABLE_SSE2   "Force-enable fisk's SSE2-accelerated code paths; only safe if the deployment target actually has SSE2" OFF)
option(FISK_ENABLE_AVX2   "Force-enable fisk's AVX2-accelerated code paths; only safe if the deployment target actually has AVX2" OFF)
option(FISK_ENABLE_AVX512 "Force-enable fisk's AVX512-accelerated code paths; only safe if the deployment target actually has AVX512" OFF)
option(FISK_ENABLE_NEON   "Force-enable fisk's NEON-accelerated code paths (ARM64); only safe if the deployment target actually has NEON" OFF)

if (FISK_ENABLE_BMI2 AND COMPILER_SUPPORTS_BMI2)
    target_compile_options(fisk INTERFACE "-mbmi2")
    target_compile_definitions(fisk INTERFACE FISK_HAS_BMI2=1)
endif()

if (FISK_ENABLE_PCLMUL AND COMPILER_SUPPORTS_PCLMUL)
    target_compile_options(fisk INTERFACE "-mpclmul")
    target_compile_definitions(fisk INTERFACE FISK_HAS_CLMUL=1)
endif()

if (FISK_ENABLE_SSE2 AND COMPILER_SUPPORTS_SSE2)
    target_compile_options(fisk INTERFACE "-msse2")
    target_compile_definitions(fisk INTERFACE FISK_HAS_SSE2=1)
endif()

if (FISK_ENABLE_AVX2 AND COMPILER_SUPPORTS_AVX2)
    target_compile_options(fisk INTERFACE "-mavx2")
    target_compile_definitions(fisk INTERFACE FISK_HAS_AVX2=1)
endif()

# AVX512BW is required alongside AVX512F: seq_pack's Msb byte-reversal uses _mm512_shuffle_epi8,
# which needs BW. It's present on essentially all mainstream AVX512 hardware since Skylake-X/Ice
# Lake -- only the earliest, now-obsolete Xeon Phi "Knights Landing" parts are F-only -- so this is
# folded into the single FISK_ENABLE_AVX512 option rather than a separate opt-in tier.
if (FISK_ENABLE_AVX512 AND COMPILER_SUPPORTS_AVX512 AND COMPILER_SUPPORTS_AVX512BW)
    target_compile_options(fisk INTERFACE "-mavx512f" "-mavx512bw")
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
            $<$<CONFIG:Release>:-O3 -march=native>
        )

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
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(fisk_benchmarks PRIVATE
            /W4
            /WX
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
        tests/test_seq_enc.cpp
        tests/test_bit_extract.cpp
        tests/test_seq_pack.cpp
    )
    target_include_directories(fisk_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(fisk_tests PRIVATE fisk::fisk)

    # Same rationale as fisk_benchmarks above: build-quality choices private to this binary.
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        # Activate whatever ISA extensions this build machine actually has, for the test
        # suite only, so that the library itself stays portable.
        target_compile_options(fisk_tests PRIVATE
            -march=native
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
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(fisk_tests PRIVATE
            /W4
            /WX
        )
    endif()

    add_test(NAME fisk_tests COMMAND fisk_tests)
endif()
