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
//     Compiler Info
// =================================================================================================

std::string info_platform();
std::string info_compiler_family();
std::string info_compiler_version();

void info_print_compiler(std::ostream& os);

// =================================================================================================
//     Hardware Info
// =================================================================================================

std::string info_cpu_vendor();
std::string info_cpu_model();

void info_print_cpu(std::ostream& os);

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
