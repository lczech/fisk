#pragma once

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

// Preprocessor checks for intrinsics support. Each of these is active either because our build
// system (see CMakeLists.txt's FISK_ENABLE_* options) explicitly defined the FISK_HAS_* flag, or
// because the compiler already has the feature active ambiently (e.g., via -march=native), in
// which case we pick that up automatically without needing any fisk-specific configuration.
// All five follow the same shape here, so that none of them is a special case to remember.
#if defined(FISK_HAS_BMI2) || defined(__BMI2__)
    #include <immintrin.h>
    #define FISK_HAS_BMI2 1
#endif
#if defined(FISK_HAS_SSE2) || defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
    #include <emmintrin.h>
    #define FISK_HAS_SSE2 1
#endif
#if defined(FISK_HAS_AVX2) || defined(__AVX2__)
    #include <immintrin.h>
    #define FISK_HAS_AVX2 1
#endif
#if defined(FISK_HAS_AVX512) || defined(__AVX512F__)
    #include <immintrin.h>
    #define FISK_HAS_AVX512 1
#endif
#if defined(FISK_HAS_NEON) || (defined(__aarch64__) && defined(__ARM_NEON)) || defined(__ARM_NEON__)
    #include <arm_neon.h>
    #define FISK_HAS_NEON 1
#endif
#if defined(_MSC_VER)
    #include <intrin.h>
#endif
