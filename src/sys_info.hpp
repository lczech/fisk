#pragma once

#include <cstdint>
#include <iosfwd>
#include <ostream>
#include <string>
#include <stdexcept>

// =================================================================================================
//     Platform Macros
// =================================================================================================

// We have code that is specific to x86 Intel BMI2 and other intrinsics,
// which we hence need to deactivate on Apple ARM, such as M2/M3 prcessors.
#if defined(__x86_64__) || defined(_M_X64)
    #define PLATFORM_X86_64 1
    #if defined(__GNUC__) || defined(__clang__)
        #define SYSTEM_X86_64_GNU_CLANG 1
    #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define PLATFORM_ARM64 1
#else
    #error "Unsupported architecture"
#endif

// Preprocessor checks for intrinsics support.
// The BMI2 is set in cmake, the other ones are checked here dynamically.
#if defined(FISK_HAS_BMI2) || defined(__BMI2__)
    #include <immintrin.h>
    #define FISK_HAS_BMI2 1
#endif
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
    #include <emmintrin.h>
    #define FISK_HAS_SSE2 1
#endif
#if defined(__AVX2__)
    #include <immintrin.h>
    #define FISK_HAS_AVX2 1
#endif
#if defined(__AVX512F__)
    #include <immintrin.h>
    #define FISK_HAS_AVX512 1
#endif
#if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(__ARM_NEON__)
    #include <arm_neon.h>
    #define FISK_HAS_NEON 1
#endif
#if defined(_MSC_VER)
    #include <intrin.h>
#endif

#include "bit_extract.hpp"
#include "seq_enc.hpp"

// =================================================================================================
//     Hardware Info
// =================================================================================================

std::string info_platform_name();
std::string info_platform_arch();
void info_print_platform(std::ostream& os);

std::string info_cpu_vendor();
std::string info_cpu_model();
void info_print_cpu(std::ostream& os);

// =================================================================================================
//     Compiler Info
// =================================================================================================

std::string info_compiler_family();
std::string info_compiler_version();
void info_print_compiler(std::ostream& os);

// =================================================================================================
//     CPU Intrinsics
// =================================================================================================

// -----------------------------------------------------------------
// Compile-time support
// -----------------------------------------------------------------

bool compiled_bmi2() noexcept;
bool compiled_sse2() noexcept;
bool compiled_avx2() noexcept;
bool compiled_avx512() noexcept;
bool compiled_neon() noexcept;

// -----------------------------------------------------------------
// Runtime CPU / OS support
// -----------------------------------------------------------------

bool cpu_bmi2() noexcept;
bool cpu_sse2() noexcept;
bool cpu_avx2() noexcept;
bool cpu_avx512() noexcept;
bool cpu_neon() noexcept;

// -----------------------------------------------------------------
// Combined checks: "safe to use in this build"
// -----------------------------------------------------------------

bool bmi2_enabled() noexcept;
bool sse2_enabled() noexcept;
bool avx2_enabled() noexcept;
bool avx512_enabled() noexcept;
bool neon_enabled() noexcept;

// -----------------------------------------------------------------
// Reporting
// -----------------------------------------------------------------

void info_print_intrinsics(std::ostream& os);
