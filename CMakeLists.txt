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

include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-mbmi2"   COMPILER_SUPPORTS_BMI2)
check_cxx_compiler_flag("-mpclmul" COMPILER_SUPPORTS_PCLMUL)

if (COMPILER_SUPPORTS_BMI2)
    target_compile_options(fisk INTERFACE "-mbmi2")
    target_compile_definitions(fisk INTERFACE FISK_HAS_BMI2=1)
endif()

if (COMPILER_SUPPORTS_PCLMUL)
    target_compile_options(fisk INTERFACE "-mpclmul")
    target_compile_definitions(fisk INTERFACE FISK_HAS_CLMUL=1)
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
    target_compile_options(fisk_benchmarks PRIVATE
        $<$<CONFIG:Release>:-O3 -march=native>
    )

    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
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
