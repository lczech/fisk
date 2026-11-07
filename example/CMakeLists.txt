# The example of fisk, set up as a program that uses fisk would: see README.md. To build it:
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
