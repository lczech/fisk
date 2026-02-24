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

# Collect all sources in src/
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")

# Add the target binary
add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(
    ${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src
)

# ==================================================================================================
#     Build Instructions
# ==================================================================================================

# Ensure a default build type if none is given
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Set optimization level for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

# Compiler warnings (GCC / Clang / MSVC)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(${PROJECT_NAME} PRIVATE
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
    target_compile_options(${PROJECT_NAME} PRIVATE
        /W4
        /WX
    )
endif()

# ==================================================================================================
#     Detect Instruction Sets
# ==================================================================================================

include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-mbmi2"   COMPILER_SUPPORTS_BMI2)
check_cxx_compiler_flag("-mpclmul" COMPILER_SUPPORTS_PCLMUL)

if (COMPILER_SUPPORTS_BMI2)
    target_compile_options(${PROJECT_NAME} PRIVATE "-mbmi2")
    target_compile_definitions(${PROJECT_NAME} PRIVATE HAVE_BMI2=1)
endif()

if (COMPILER_SUPPORTS_PCLMUL)
    target_compile_options(${PROJECT_NAME} PRIVATE "-mpclmul")
    target_compile_definitions(${PROJECT_NAME} PRIVATE HAVE_CLMUL=1)
endif()
