#pragma once

#include <atomic>
#include <bit>
#include <cstdint>
#include <type_traits>

#include "fisk/core/platform.hpp"

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

// Compiler attribute macros, centralized so that each is spelled out exactly once. Both
// [[gnu::...]] and [[msvc::...]] are standard attribute syntax (unlike __forceinline /
// __declspec(noinline), they need no special placement relative to `inline`), so branching the
// macro definition alone is enough -- no call site needs to change per compiler.
#if defined(_MSC_VER)
    #define FISK_ALWAYS_INLINE [[msvc::forceinline]]
#elif defined(__GNUC__) || defined(__clang__)
    #define FISK_ALWAYS_INLINE [[gnu::always_inline]]
#else
    #define FISK_ALWAYS_INLINE
#endif

// Alias for FISK_ALWAYS_INLINE, applied to the for_each_kmer_*/for_each_spaced_kmer_* driver
// loops specifically. Distinguishes "always-inlined because it's one of the hot per-sequence
// loops" from any other reason a function might carry FISK_ALWAYS_INLINE, and gives that family
// a single place to pick up something more (e.g. [[gnu::flatten]]) later, if ever needed.
#define FISK_ALWAYS_INLINE_FOR_EACH FISK_ALWAYS_INLINE

// For rare/cold-path helpers (precondition checks that throw, tail-case loops) that should stay
// out of a hot path's inlined body and icache footprint. MSVC has no "cold" hint to pair with
// noinline, so it only gets the noinline half.
#if defined(_MSC_VER)
    #define FISK_NOINLINE_COLD [[msvc::noinline]]
#elif defined(__GNUC__) || defined(__clang__)
    #define FISK_NOINLINE_COLD [[gnu::noinline, gnu::cold]]
#else
    #define FISK_NOINLINE_COLD
#endif

namespace fisk {

// =================================================================================================
//     Byte Swap
// =================================================================================================

// Needed e.g. to convert between the two PackedSequence layouts (see core/types.hpp).
//
// Tiered fallback, fastest/most standard first:
//   1) std::byteswap (C++23; we currently use C++20, so this only activates once building
//      with a standard library that has shipped it, tracked via the __cpp_lib_byteswap
//      feature-test macro rather than checking the language mode directly).
//   2) The MSVC/GCC/Clang compiler builtins. MSVC's are not constexpr, so that branch routes
//      constant evaluation through the portable form instead, via std::is_constant_evaluated().
//      That keeps all three tiers usable in constant expressions.
//   3) A fully portable hand-rolled shift-mask-or fallback. This only collapses to a single
//      instruction once the optimizer's idiom-recognition kicks in (reliably the case for GCC,
//      Clang, and MSVC at -O2/-O3), kept as last resort.
#if defined(__cpp_lib_byteswap) && __cpp_lib_byteswap >= 202110L

    inline constexpr std::uint16_t byte_swap_16(std::uint16_t x) noexcept
    {
        return std::byteswap(x);
    }

    inline constexpr std::uint32_t byte_swap_32(std::uint32_t x) noexcept
    {
        return std::byteswap(x);
    }

    inline constexpr std::uint64_t byte_swap_64(std::uint64_t x) noexcept
    {
        return std::byteswap(x);
    }

#elif defined(_MSC_VER)

    inline constexpr std::uint16_t byte_swap_16(std::uint16_t x) noexcept
    {
        if (std::is_constant_evaluated()) {
            return static_cast<std::uint16_t>((x << 8) | (x >> 8));
        }
        return _byteswap_ushort(x);
    }

    inline constexpr std::uint32_t byte_swap_32(std::uint32_t x) noexcept
    {
        if (std::is_constant_evaluated()) {
            x = ((x & 0x0000FFFFu) << 16) | ((x & 0xFFFF0000u) >> 16);
            x = ((x & 0x00FF00FFu) <<  8) | ((x & 0xFF00FF00u) >>  8);
            return x;
        }
        return _byteswap_ulong(x);
    }

    inline constexpr std::uint64_t byte_swap_64(std::uint64_t x) noexcept
    {
        if (std::is_constant_evaluated()) {
            x = ((x & 0x00000000FFFFFFFFULL) << 32) | ((x & 0xFFFFFFFF00000000ULL) >> 32);
            x = ((x & 0x0000FFFF0000FFFFULL) << 16) | ((x & 0xFFFF0000FFFF0000ULL) >> 16);
            x = ((x & 0x00FF00FF00FF00FFULL) <<  8) | ((x & 0xFF00FF00FF00FF00ULL) >>  8);
            return x;
        }
        return _byteswap_uint64(x);
    }

#elif defined(__GNUC__) || defined(__clang__)

    inline constexpr std::uint16_t byte_swap_16(std::uint16_t x) noexcept
    {
        return __builtin_bswap16(x);
    }

    inline constexpr std::uint32_t byte_swap_32(std::uint32_t x) noexcept
    {
        return __builtin_bswap32(x);
    }

    inline constexpr std::uint64_t byte_swap_64(std::uint64_t x) noexcept
    {
        return __builtin_bswap64(x);
    }

#else

    inline constexpr std::uint16_t byte_swap_16(std::uint16_t x) noexcept
    {
        return static_cast<std::uint16_t>((x << 8) | (x >> 8));
    }

    inline constexpr std::uint32_t byte_swap_32(std::uint32_t x) noexcept
    {
        x = ((x & 0x0000FFFFu) << 16) | ((x & 0xFFFF0000u) >> 16);
        x = ((x & 0x00FF00FFu) <<  8) | ((x & 0xFF00FF00u) >>  8);
        return x;
    }

    inline constexpr std::uint64_t byte_swap_64(std::uint64_t x) noexcept
    {
        x = ((x & 0x00000000FFFFFFFFULL) << 32) | ((x & 0xFFFFFFFF00000000ULL) >> 32);
        x = ((x & 0x0000FFFF0000FFFFULL) << 16) | ((x & 0xFFFF0000FFFF0000ULL) >> 16);
        x = ((x & 0x00FF00FF00FF00FFULL) <<  8) | ((x & 0xFF00FF00FF00FF00ULL) >>  8);
        return x;
    }

#endif

// =================================================================================================
//     Compiler Optimization Barrier
// =================================================================================================

// Memory barriers that force a value into a register (not into memory necessarily).
// Used to keep microbenchmarks honest: without it, a compiler might be able to prove a branch-free
// computation to be reducible and silently skip real work, skewing the benchmark.
// Offered for all integer widths, so that a value already held in a sub-register (e.g. the
// single-byte result of a char encoder) does not need a zero-extend just to fit the uint64_t
// overload; the "r" constraint binds directly to a register of the argument's own width.

FISK_ALWAYS_INLINE
inline void do_not_optimize(std::uint8_t v)
{
    #if defined(__GNUC__) || defined(__clang__)
        asm volatile("" : : "r"(v));
    #else
        volatile std::uint8_t sink = v;
        (void) sink;
    #endif
}

FISK_ALWAYS_INLINE
inline void do_not_optimize(std::uint16_t v)
{
    #if defined(__GNUC__) || defined(__clang__)
        asm volatile("" : : "r"(v));
    #else
        volatile std::uint16_t sink = v;
        (void) sink;
    #endif
}

FISK_ALWAYS_INLINE
inline void do_not_optimize(std::uint32_t v)
{
    #if defined(__GNUC__) || defined(__clang__)
        asm volatile("" : : "r"(v));
    #else
        volatile std::uint32_t sink = v;
        (void) sink;
    #endif
}

FISK_ALWAYS_INLINE
inline void do_not_optimize(std::uint64_t v)
{
    #if defined(__GNUC__) || defined(__clang__)
        asm volatile("" : : "r"(v));
    #else
        volatile std::uint64_t sink = v;
        (void) sink;
    #endif
}

// Overloads of the above for SIMD register types, one per register class -- unlike the scalar
// GPR case, there is no single inline-asm constraint letter that works for all of them (legacy
// SSE/AVX xmm/ymm registers, AVX-512 zmm registers, and ARM NEON registers are each their own
// constraint class), so this is an overload set rather than a template. Guarded by the same
// FISK_HAS_* macros as the types themselves, so each overload only exists where its type does.
#if defined(FISK_HAS_SSE2)
    FISK_ALWAYS_INLINE
    inline void do_not_optimize(__m128i v)
    {
        #if defined(__GNUC__) || defined(__clang__)
            asm volatile("" : : "x"(v));
        #else
            volatile __m128i sink = v;
            (void) sink;
        #endif
    }
#endif

#if defined(FISK_HAS_AVX2)
    FISK_ALWAYS_INLINE
    inline void do_not_optimize(__m256i v)
    {
        #if defined(__GNUC__) || defined(__clang__)
            asm volatile("" : : "x"(v));
        #else
            volatile __m256i sink = v;
            (void) sink;
        #endif
    }
#endif

#if defined(FISK_HAS_AVX512)
    FISK_ALWAYS_INLINE
    inline void do_not_optimize(__m512i v)
    {
        #if defined(__GNUC__) || defined(__clang__)
            // "v" (rather than "x") is the constraint for the extended EVEX-encodable vector
            // registers (zmm0-31) that a __m512i needs.
            asm volatile("" : : "v"(v));
        #else
            volatile __m512i sink = v;
            (void) sink;
        #endif
    }
#endif

#if defined(FISK_HAS_NEON)
    FISK_ALWAYS_INLINE
    inline void do_not_optimize(uint64x2_t v)
    {
        #if defined(__GNUC__) || defined(__clang__)
            // "w" is the AArch64 constraint for the SIMD&FP register file NEON vectors live in.
            asm volatile("" : : "w"(v));
        #else
            volatile uint64x2_t sink = v;
            (void) sink;
        #endif
    }
#endif

// Barrier on memory rather than on a value: tells the compiler that everything reachable through
// memory may have been read and written here. Any value loaded before this point has to be loaded
// again afterwards, and so any computation derived from such a load has to be redone.
//
// This is what do_not_optimize() above cannot do: that one only demands that *a* value ends up in
// a register, and says nothing about how the value was derived, so a computation the compiler can
// prove is repeated with unchanged inputs may still be done once and reused. Benchmarking a kernel
// over the same buffer repeatedly needs this stronger guarantee to keep every repetition real.
//
// The pointer operand does not restrict the barrier: The "memory" clobber is global either way,
// it just names the buffer the caller means, so the intent is readable at the call site.
FISK_ALWAYS_INLINE
inline void clobber_memory(void const* ptr)
{
    #if defined(__GNUC__) || defined(__clang__)
        asm volatile("" : : "g"(ptr) : "memory");
    #else
        (void) ptr;
        std::atomic_signal_fence(std::memory_order_acq_rel);
    #endif
}

} // namespace fisk
