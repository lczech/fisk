# The example of fisk, set up as a standalone program. See README.md for details.
#
# To build it, from the main directory of the repository, run:
#
#     cmake -S example -B build-example
#     cmake --build build-example

cmake_minimum_required(VERSION 3.20)
project(fisk_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# fisk, from the folder above. In your own project, see the README of fisk for FetchContent.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR}/fisk)

add_executable(fisk_example main.cpp)
target_link_libraries(fisk_example PRIVATE fisk::fisk)

# The example runs on the computer that builds it, so it may use all instruction sets of its CPU.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-march=native FISK_EXAMPLE_MARCH_NATIVE)
if(FISK_EXAMPLE_MARCH_NATIVE)
    target_compile_options(fisk_example PRIVATE -march=native)
endif()

# Optional, for the most stable speed: start every function and loop at a 64-byte boundary, so that
# the speed of a hot loop does not depend on where the code around it puts it. The benchmarks of
# fisk use the same flags. A program can leave them out; the speed of its loops can then differ
# between builds.
check_cxx_compiler_flag(-falign-loops=64 FISK_EXAMPLE_ALIGN_LOOPS)
if(FISK_EXAMPLE_ALIGN_LOOPS)
    target_compile_options(fisk_example PRIVATE -falign-functions=64 -falign-loops=64)
endif()
