#pragma once

#include <iostream>
#include <string>
#include <stdexcept>

#ifdef HAVE_BMI2
#include <immintrin.h>
#endif

#ifdef HAVE_CLMUL
#include <wmmintrin.h>
#endif

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

inline bool bmi2_enabled()
{
    // Cached result for speed
    static bool const enabled = [] {
        bool compiled = false;
        #ifdef HAVE_BMI2
            compiled = true;
        #endif

        bool cpu = false;
        #if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
            __builtin_cpu_init();
            cpu = __builtin_cpu_supports("bmi2");
        #endif

        return compiled && cpu;
    }();

    return enabled;
}

inline bool clmul_enabled()
{
    // Cached result for speed
    static bool const enabled = [] {
        bool compiled = false;
        #ifdef HAVE_CLMUL
            compiled = true;
        #endif

        bool cpu = false;
        #if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
            __builtin_cpu_init();
            cpu = __builtin_cpu_supports("pclmul");
        #endif

        return compiled && cpu;
    }();

    return enabled;
}

void info_print_intrinsics(std::ostream& os);
