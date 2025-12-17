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

inline void print_intrinsics_support()
{
    std::cout << "Instruction set support:\n";

    // -----------------------------
    // BMI2
    // -----------------------------
    bool cmake_bmi2 =
    #ifdef HAVE_BMI2
        true;
    #else
        false;
    #endif

    bool cpu_bmi2 = false;
    #if defined(__GNUC__) || defined(__clang__)
        __builtin_cpu_init();
        cpu_bmi2 = __builtin_cpu_supports("bmi2");
    #endif

    std::cout << "  BMI2  : compiled=" << (cmake_bmi2 ? "yes, " : "no,  ")
              << "cpu=" << (cpu_bmi2 ? "yes" : "no") << "\n";

    // -----------------------------
    // CLMUL (PCLMUL)
    // -----------------------------
    bool cmake_clmul =
    #ifdef HAVE_CLMUL
        true;
    #else
        false;
    #endif

    bool cpu_clmul = false;
    #if defined(__GNUC__) || defined(__clang__)
        __builtin_cpu_init();
        cpu_clmul = __builtin_cpu_supports("pclmul");
    #endif

    std::cout << "  CLMUL : compiled=" << (cmake_clmul ? "yes, " : "no,  ")
              << "cpu=" << (cpu_clmul ? "yes" : "no") << "\n";
}
