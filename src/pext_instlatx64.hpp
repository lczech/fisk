#pragma once

// The following implementation is from:
//
// InstLatX64_Demo
// https://github.com/InstLatx64/InstLatX64_Demo
// https://github.com/InstLatx64/InstLatX64_Demo/blob/795b20a2b2dd783772052bbde4b19e5418510d53/PEXT_PDEP_Emu.cpp
// Downloaded on 2025-12-17
//
// We mostly reproduce the code here as-is, with several fixes for (un)signed integer conversion.
// Furthermore, we change the types `__int64` and `unsigned __int64` to the proper C++ ones,
// instead of their MSVC versions, which do not work on other platforms.
// Similarly, we replace several MSCV-specific functions with portable wrappers and replacements.
// Furthermore, we add `inline` to all functions here, to allow more compiler optimizations.
// We also clean up a few preprocessor definitions, as well as an unfortunate `using namespace std`,
// and replace pop count intrinsics with the C++20 std::popcount().


// The whole functionality relies heavily on Intel intrinsics, which are not available on ARM.
#include "sys_info.hpp"
#ifdef PLATFORM_X86_64

#include <immintrin.h>
#include <wmmintrin.h>
#include <cstdint>
#include <cassert>
#include <bit>

constexpr std::uint32_t bextr_u32(
    std::uint32_t x,
    unsigned start,
    unsigned len
) noexcept  {
    // Matches _bextr_u32 semantics:
    // extract `len` bits starting at `start`, right-aligned.
    if (len == 0) {
        return 0u;
    }
    if (len >= 32) {
        return x >> start;
    }
    return (x >> start) & ((std::uint32_t{1} << len) - 1u);
}

constexpr std::uint64_t bextr_u64(
    std::uint64_t x,
    unsigned start,
    unsigned len
) noexcept  {
    // Matches _bextr_u64 semantics:
    // extract `len` bits starting at `start`, right-aligned.
    if (len == 0) {
        return 0;
    }
    if (len >= 64) {
        return x >> start;
    }
    return (x >> start) & ((std::uint64_t{1} << len) - 1);
}

constexpr std::uint32_t blsr_u32(std::uint32_t x) noexcept
{
    return x & (x - 1);
}

constexpr std::uint64_t blsr_u64(std::uint64_t x) noexcept
{
    return x & (x - 1);
}

// =================================================================================================
//     Original source
// =================================================================================================

// #include "stdafx.h"
// #include "PEXT_PDEP_Emu.h"

//Credit: Zach Wegner
//Based on https://github.com/zwegner/zp7 project with unrolling and sparse case handling

// using namespace std;

inline unsigned int pext32_emu(unsigned int v, unsigned int m)
{
    unsigned int ret = 0, pc = static_cast<unsigned int>(std::popcount(m));
    switch (pc) {
        case 0:
            ret = 0;
            break;
        case 1:
            ret = (v & m) != 0;
            break;
        case 2: {
                unsigned int msb = bextr_u32(v, (static_cast<unsigned int>(31 - std::countl_zero(m))), 1);
                unsigned int lsb = bextr_u32(v, static_cast<unsigned int>(std::countr_zero(m)), 1);
                ret = (msb << 1) | lsb;
            }
           break;
        case 3: {
                const unsigned int lz = static_cast<unsigned int>(31 - std::countl_zero(m));
                const unsigned int tz = static_cast<unsigned int>(std::countr_zero(m));
                unsigned int msb = bextr_u32(v, lz, 1);
                unsigned int lsb = bextr_u32(v, tz, 1);
                m = blsr_u32(m);
                unsigned int csb = bextr_u32(v, static_cast<unsigned int>(std::countr_zero(m)), 1);
                ret = (msb << 2) | (csb << 1) | lsb;
            }
            break;
        case 4: {
                const unsigned int lz = static_cast<unsigned int>(31 - std::countl_zero(m));
                const unsigned int tz = static_cast<unsigned int>(std::countr_zero(m));
                unsigned int msb1 = bextr_u32(v, lz, 1);
                unsigned int lsb1 = bextr_u32(v, tz, 1);
                m &= ~((1u << lz) | (1u << tz));
                unsigned int msb0 = bextr_u32(v, static_cast<unsigned int>(31 - std::countl_zero(m)), 1);
                unsigned int lsb0 = bextr_u32(v, static_cast<unsigned int>(std::countr_zero(m)), 1);
                ret = (msb1 << 3) | (msb0 << 2) | (lsb0 << 1) | lsb1;
            break;
        }
        //case 5:
        //case 6: {
        //		unsigned int lsb = 0, msb = 0;
        //		do {
        //		const unsigned int lz = static_cast<unsigned int>(31 - std::countl_zero(m));
        //		const unsigned int tz = std::countr_zero(m);
        //			lsb = _rotr(lsb | bextr_u32(v, tz, 0x1), 1);
        //			msb = ((msb << 1) | bextr_u32(v, lz, 0x1));
        //			m &= ~((1u << lz) | (1u << tz));
        //		} while (m);
        //		ret = _rotl((lsb << (pc & 1)) | msb, (pc >> 1));
        //	} break;
        default: {
            __m128i mm		= _mm_cvtsi32_si128(static_cast<int>(~m));
            __m128i mtwo	= _mm_set1_epi64x(static_cast<std::int64_t>((~0ULL) - 1));
            __m128i clmul	= _mm_clmulepi64_si128(mm, mtwo, 0);
    unsigned int	bit0	= static_cast<unsigned int>(_mm_cvtsi128_si32(clmul));
    unsigned int	a		= v & m;
                    a		= (~bit0 & a) | ((bit0 & a) >> 1);
                    mm		= _mm_and_si128(mm, clmul);
                    clmul	= _mm_clmulepi64_si128(mm, mtwo, 0);
    unsigned int	bit1	= static_cast<unsigned int>(_mm_cvtsi128_si32(clmul));
                    a		= (~bit1 & a) | ((bit1 & a) >> 2);
                    mm		= _mm_and_si128(mm, clmul);
                    clmul	= _mm_clmulepi64_si128(mm, mtwo, 0);
    unsigned int	bit2	= static_cast<unsigned int>(_mm_cvtsi128_si32(clmul));
                    a		= (~bit2 & a) | ((bit2 & a) >> 4);
                    mm		= _mm_and_si128(mm, clmul);
                    clmul	= _mm_clmulepi64_si128(mm, mtwo, 0);
    unsigned int	bit3	= static_cast<unsigned int>(_mm_cvtsi128_si32(clmul));
                    a		= (~bit3 & a) | ((bit3 & a) >> 8);
                    mm		= _mm_and_si128(mm, clmul);
                    clmul	= _mm_sub_epi64(_mm_setzero_si128(), mm);
    unsigned int	bit4	= static_cast<unsigned int>(_mm_cvtsi128_si32(clmul));
                    bit4	+= bit4;
                    ret		= (unsigned int)((~bit4 & a) | ((bit4 & a) >> 16));
            break;
        }
    }
    return ret;
};

inline unsigned int pdep32_emu(unsigned int v, unsigned int m)
{
    unsigned int ret = 0, pc = static_cast<unsigned int>(std::popcount(m));
    switch (pc) {
        case 0:
            ret = 0;
            break;
        case 1:
            ret = (v & 1) << std::countr_zero(m);
            break;
        case 2:
            ret = (((v << (32 - pc)) & 0x80000000) >> std::countl_zero(m)) | ((v & 1) << std::countr_zero(m));
            break;
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 10:
        case 11:
        case 12:
        case 13: {
            unsigned int lsb = 0, msb = 0;
            unsigned int v1 = v << (32 - pc);
            for (unsigned int i = 0; i < pc / 2  ; i++) {
                const unsigned int tz = static_cast<unsigned int>(std::countr_zero(m));
                const unsigned int lz = static_cast<unsigned int>(std::countl_zero(m));
                m &= ~((0x80000000 >> lz) | (1u << tz));
                msb = (v1 & 0x80000000) >> lz;
                lsb = (v & 1) << tz;
                ret |= (msb | lsb);
                v >>= 1;
                v1 <<= 1;
            }
            ret |= ((pc & 1) & v) << std::countr_zero(m);
            break;
        }
        default: {
            __m128i mtwo	= _mm_set1_epi64x(static_cast<std::int64_t>((~0ULL) - 1));
            __m128i mm		= _mm_cvtsi32_si128(static_cast<int>(~m));
            __m128i bit0	= _mm_clmulepi64_si128(mm, mtwo, 0);
                    mm		= _mm_and_si128(mm, bit0);
            __m128i bit1	= _mm_clmulepi64_si128(mm, mtwo, 0);
                    mm		= _mm_and_si128(mm, bit1);
            __m128i bit2	= _mm_clmulepi64_si128(mm, mtwo, 0);
                    mm		= _mm_and_si128(mm, bit2);
            __m128i bit3	= _mm_clmulepi64_si128(mm, mtwo, 0);
                    mm		= _mm_and_si128(mm, bit3);
            __m128i bit4	= _mm_sub_epi64(_mm_setzero_si128(), mm);
                    bit4	= _mm_add_epi64(bit4, bit4);
            __m128i a		= _mm_cvtsi32_si128(static_cast<int>(_bzhi_u32(v, pc)));

                    bit4	= _mm_srli_epi64(bit4, 16);
                    a		= _mm_add_epi64(_mm_andnot_si128(bit4, a),_mm_slli_epi64(_mm_and_si128(bit4, a), 16));
                    bit3	= _mm_srli_epi64(bit3, 8);
                    a		= _mm_add_epi64(_mm_andnot_si128(bit3, a),_mm_slli_epi64(_mm_and_si128(bit3, a), 8));
                    bit2	= _mm_srli_epi64(bit2, 4);
                    a		= _mm_add_epi64(_mm_andnot_si128(bit2, a),_mm_slli_epi64(_mm_and_si128(bit2, a), 4));
                    bit1	= _mm_srli_epi64(bit1, 2);
                    a		= _mm_add_epi64(_mm_andnot_si128(bit1, a),_mm_slli_epi64(_mm_and_si128(bit1, a), 2));
                    bit0	= _mm_srli_epi64(bit0, 1);
                    a		= _mm_add_epi64(_mm_andnot_si128(bit0, a),_mm_slli_epi64(_mm_and_si128(bit0, a), 1));
            ret = static_cast<unsigned int>(_mm_cvtsi128_si32(a));
        }
        break;
    }
    return ret;
};

inline std::uint64_t pext64_emu(std::uint64_t v, std::uint64_t m)
{
    std::uint64_t ret = 0;
    unsigned int pc = static_cast<unsigned int>(std::popcount(m));
    switch (pc) {
        case 0:
            ret = 0;
            break;
        case 1:
            ret = (v & m) != 0;
            break;
        case 2: {
                std::uint64_t msb = bextr_u64(v, (unsigned int)(63 - std::countl_zero(m)), 1);
                std::uint64_t lsb = bextr_u64(v, (unsigned int)std::countr_zero(m), 1);
                ret = (msb << 1) | lsb;
            }
            break;
        case 3: {
                std::uint64_t msb = bextr_u64(v, (unsigned int)(63 - std::countl_zero(m)), 1);
                std::uint64_t lsb = bextr_u64(v, (unsigned int)std::countr_zero(m), 1);
                m = blsr_u64(m);
                std::uint64_t csb = bextr_u64(v, (unsigned int)std::countr_zero(m), 1);
                ret = (msb << 2) | (csb << 1) | lsb;
            }
            break;
        case 4: {
                const unsigned int lz1 = (unsigned int)(63 - std::countl_zero(m));
                const unsigned int tz1 = (unsigned int)std::countr_zero(m);
                m &= ~((1ULL << lz1) | (1ULL << tz1));
                std::uint64_t msb1 = bextr_u64(v, lz1, 1);
                std::uint64_t lsb1 = bextr_u64(v, tz1, 1);
                ret = (msb1 << 3) | lsb1;
                std::uint64_t msb0 = bextr_u64(v, (unsigned int)(63 - std::countl_zero(m)), 1);
                std::uint64_t lsb0 = bextr_u64(v, (unsigned int)std::countr_zero(m), 1);
                ret |= (msb0 << 2) | (lsb0 << 1);
            break;
        }
        case 5: {
                const unsigned int lz1 = (unsigned int)(63 - std::countl_zero(m));
                const unsigned int tz1 = (unsigned int)std::countr_zero(m);
                m &= ~((1ULL << lz1) | (1ULL << tz1));
                std::uint64_t msb1 = bextr_u64(v, lz1, 1);
                std::uint64_t lsb1 = bextr_u64(v, tz1, 1);
                const unsigned int lz0 = (unsigned int)(63 - std::countl_zero(m));
                const unsigned int tz0 = (unsigned int)std::countr_zero(m);
                m &= ~((1ULL << lz0) | (1ULL << tz0));
                ret = (msb1 << 4) | lsb1;
                std::uint64_t msb0 = bextr_u64(v, lz0, 1);
                std::uint64_t lsb0 = bextr_u64(v, tz0, 1);
                ret |= (msb0 << 3) | (lsb0 << 1);
                std::uint64_t csb = bextr_u64(v, (unsigned int)std::countr_zero(m), 1);
                ret |= csb << 2;
            break;
        }
        case 6: {
                const unsigned int lz2 = (unsigned int)(63 - std::countl_zero(m));
                const unsigned int tz2 = (unsigned int)std::countr_zero(m);
                std::uint64_t msb2 = bextr_u64(v, lz2, 1);
                std::uint64_t lsb2 = bextr_u64(v, tz2, 1);
                m &= ~((1ULL << lz2) | (1ULL << tz2));
                const unsigned int lz1 = (unsigned int)(63 - std::countl_zero(m));
                const unsigned int tz1 = (unsigned int)std::countr_zero(m);
                ret = (msb2 << 5) | lsb2;
                std::uint64_t msb1 = bextr_u64(v, lz1, 1);
                std::uint64_t lsb1 = bextr_u64(v, tz1, 1);
                m &= ~((1ULL << lz1) | (1ULL << tz1));
                ret |= (msb1 << 4) | (lsb1 << 1);
                std::uint64_t msb0 = bextr_u64(v, (unsigned int)(63 - std::countl_zero(m)), 1);
                std::uint64_t lsb0 = bextr_u64(v, (unsigned int)std::countr_zero(m), 1);
                ret |= (msb0 << 3) | (lsb0 << 2);
            break;
        }
        case 7: {
                const unsigned int lz2 = (unsigned int)(63 - std::countl_zero(m));
                const unsigned int tz2 = (unsigned int)std::countr_zero(m);
                std::uint64_t msb2 = bextr_u64(v, lz2, 1);
                std::uint64_t lsb2 = bextr_u64(v, tz2, 1);
                m &= ~((1ULL << lz2) | (1ULL << tz2));
                const unsigned int lz1 = (unsigned int)(63 - std::countl_zero(m));
                const unsigned int tz1 = (unsigned int)std::countr_zero(m);
                ret = (msb2 << 6) | lsb2;
                std::uint64_t msb1 = bextr_u64(v, lz1, 1);
                std::uint64_t lsb1 = bextr_u64(v, tz1, 1);
                m &= ~((1ULL << lz1) | (1ULL << tz1));
                const unsigned int lz0 = (unsigned int)(63 - std::countl_zero(m));
                const unsigned int tz0 = (unsigned int)std::countr_zero(m);
                ret |= (msb1 << 5) | (lsb1 << 1);
                std::uint64_t msb0 = bextr_u64(v, lz0, 1);
                std::uint64_t lsb0 = bextr_u64(v, tz0, 1);
                m &= ~((1ULL << lz0) | (1ULL << tz0));
                ret |= (msb0 << 4) | (lsb0 << 2);
                std::uint64_t csb = bextr_u64(v, (unsigned int)std::countr_zero(m), 1);
                ret |= csb << 3;
            break;
        }
        //case 5:
        //case 6: {
        //		std::uint64_t lsb = 0, msb = 0;
        //		do {
        //			const unsigned int lz = (unsigned int)(63 - std::countl_zero(m));
        //			const unsigned int tz = (unsigned int)std::countr_zero(m);
        //			lsb = _rotr64(lsb | bextr_u64(v, tz, 0x1), 1);
        //			msb = ((msb << 1) | bextr_u64(v, lz, 0x1));
        //			m &= ~((1ULL << lz) | (1ULL << tz));
        //		} while (m);
        //		ret = _rotl64((lsb << (pc & 1)) | msb, (pc >> 1));
        //	} break;
        default: {
            __m128i mm		= _mm_cvtsi64_si128(static_cast<std::int64_t>(~m));
            __m128i mtwo	= _mm_set1_epi64x(static_cast<std::int64_t>((~0ULL) - 1));
            __m128i clmul	= _mm_clmulepi64_si128(mm, mtwo, 0);
    std::uint64_t bit0	= static_cast<std::uint64_t>(_mm_cvtsi128_si64(clmul));
    std::uint64_t a		= v & m;
                    a		= (~bit0 & a) | ((bit0 & a) >> 1);
                    mm		= _mm_and_si128(mm, clmul);
                    clmul	= _mm_clmulepi64_si128(mm, mtwo, 0);
    std::uint64_t bit1	= static_cast<std::uint64_t>(_mm_cvtsi128_si64(clmul));
                    a		= (~bit1 & a) | ((bit1 & a) >> 2);
                    mm		= _mm_and_si128(mm, clmul);
                    clmul	= _mm_clmulepi64_si128(mm, mtwo, 0);
    std::uint64_t bit2	= static_cast<std::uint64_t>(_mm_cvtsi128_si64(clmul));
                    a		= (~bit2 & a) | ((bit2 & a) >> 4);
                    mm		= _mm_and_si128(mm, clmul);
                    clmul	= _mm_clmulepi64_si128(mm, mtwo, 0);
    std::uint64_t bit3	= static_cast<std::uint64_t>(_mm_cvtsi128_si64(clmul));
                    a		= (~bit3 & a) | ((bit3 & a) >> 8);
                    mm		= _mm_and_si128(mm, clmul);
                    clmul	= _mm_clmulepi64_si128(mm, mtwo, 0);
    std::uint64_t bit4	= static_cast<std::uint64_t>(_mm_cvtsi128_si64(clmul));
                    a		= (~bit4 & a) | ((bit4 & a) >> 16);
                    mm		= _mm_and_si128(mm, clmul);
                    clmul	= _mm_sub_epi64(_mm_setzero_si128(), mm);
    std::uint64_t bit5	= static_cast<std::uint64_t>(_mm_cvtsi128_si64(clmul));
                    bit5	+= bit5;
                    ret		= (~bit5 & a) | ((bit5 & a) >> 32);
        } break;
    }
    return ret;
};

inline std::uint64_t pdep64_emu(std::uint64_t v, std::uint64_t m)
{
    unsigned int pc = static_cast<unsigned int>(std::popcount(m));
    std::uint64_t ret = 0;
    switch (pc) {
        case 0:
                ret = 0;
            break;
        case 1:
                ret = (v & 1) << std::countr_zero(m);
            break;
        case 2:
                ret = (((v << (64 - pc)) & 0x8000000000000000) >> std::countl_zero(m)) | ((v & 1) << std::countr_zero(m));
            break;
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
        case 15: {
                std::uint64_t lsb = 0, msb = 0;
                std::uint64_t v1 = v << (64 - pc);
                for (unsigned int i = 0; i < pc / 2; i++) {
                    const std::uint64_t tz = static_cast<std::uint64_t>(std::countr_zero(m));
                    const std::uint64_t lz = static_cast<std::uint64_t>(std::countl_zero(m));
                    m &= ~((0x8000000000000000 >> lz) | (1ULL << tz));
                    msb = (v1 & 0x8000000000000000) >> lz;
                    lsb = (v & 1) << tz;
                    ret |= (msb | lsb);
                    v >>= 1;
                    v1 <<= 1;
                }
                ret |= ((pc & 1) & v) << std::countr_zero(m);
            } break;
        default: {
            __m128i mtwo	= _mm_set1_epi64x(static_cast<std::int64_t>((~0ULL) - 1));
            __m128i mm		= _mm_cvtsi64_si128(static_cast<std::int64_t>(~m));
            __m128i bit0	= _mm_clmulepi64_si128(mm, mtwo, 0);
                    mm		= _mm_and_si128(mm, bit0);
            __m128i bit1	= _mm_clmulepi64_si128(mm, mtwo, 0);
                    mm		= _mm_and_si128(mm, bit1);
            __m128i bit2	= _mm_clmulepi64_si128(mm, mtwo, 0);
                    mm		= _mm_and_si128(mm, bit2);
            __m128i bit3	= _mm_clmulepi64_si128(mm, mtwo, 0);
                    mm		= _mm_and_si128(mm, bit3);
            __m128i bit4	= _mm_clmulepi64_si128(mm, mtwo, 0);
                    mm		= _mm_and_si128(mm, bit4);
            __m128i bit5	= _mm_sub_epi64(_mm_setzero_si128(), mm);
                    bit5	= _mm_add_epi64(bit5, bit5);
            __m128i a		= _mm_cvtsi64_si128(static_cast<std::int64_t>(_bzhi_u64(v, pc)));

                    bit5	= _mm_srli_epi64(bit5, 32);
                    a		= _mm_add_epi64(_mm_andnot_si128(bit5, a),_mm_slli_epi64(_mm_and_si128(bit5, a), 32));
                    bit4	= _mm_srli_epi64(bit4, 16);
                    a		= _mm_add_epi64(_mm_andnot_si128(bit4, a),_mm_slli_epi64(_mm_and_si128(bit4, a), 16));
                    bit3	= _mm_srli_epi64(bit3, 8);
                    a		= _mm_add_epi64(_mm_andnot_si128(bit3, a),_mm_slli_epi64(_mm_and_si128(bit3, a), 8));
                    bit2	= _mm_srli_epi64(bit2, 4);
                    a		= _mm_add_epi64(_mm_andnot_si128(bit2, a),_mm_slli_epi64(_mm_and_si128(bit2, a), 4));
                    bit1	= _mm_srli_epi64(bit1, 2);
                    a		= _mm_add_epi64(_mm_andnot_si128(bit1, a),_mm_slli_epi64(_mm_and_si128(bit1, a), 2));
                    bit0	= _mm_srli_epi64(bit0, 1);
                    a		= _mm_add_epi64(_mm_andnot_si128(bit0, a),_mm_slli_epi64(_mm_and_si128(bit0, a), 1));
            ret = static_cast<std::uint64_t>(_mm_cvtsi128_si64(a));
        }
        break;
    }
    return ret;
};

inline void PEXT_PDEP_Emu_Test()
{
    // Add constants for all-ones with the right bit width.
    auto const all_32 = ~static_cast<std::uint32_t>(0);
    auto const all_64 = ~static_cast<std::uint64_t>(0);

    std::cout << "-----------------------------------" << std::endl;
    for (unsigned int b = 0; b <= 32; b++) {
        // unsigned int x = (((1UL) << b) + ((1UL << 31) >> b)) | (1UL << 16);
        std::uint32_t x =
            ( (b == 32 ? 0u : (std::uint32_t{1} << b)) +
            ((std::uint32_t{1} << 31) >> b) ) |
            (std::uint32_t{1} << 16);
        unsigned int y = _bzhi_u32(all_32, b);
        std::cout
            << std::setw(3) << std::setfill('0') << b << " "
            << "x:0x"    << std::hex << std::setw(8) << std::setfill('0') << x
            << " orig:0x" << std::setw(8) << _pext_u32(all_32, x)
            << " emu:0x"  << std::setw(8) << pext32_emu(all_32, x)
            << " | y:0x"  << std::setw(8) << y
            << " orig:0x" << std::setw(8) << _pext_u32(all_32, y)
            << " emu:0x"  << std::setw(8) << pext32_emu(all_32, y)
            << std::dec << "\n";
        assert(_pext_u32(all_32, x) == pext32_emu(all_32, x));
        assert(_pext_u32(all_32, y) == pext32_emu(all_32, y));
    }

    std::cout << "-----------------------------------" << std::endl;
    for (unsigned int b = 0; b <= 32; b++) {
        // unsigned int x = (((1UL) << b) + ((1UL << 31) >> b)) | (1UL << 16);
        std::uint32_t x =
            ( (b == 32 ? 0u : (std::uint32_t{1} << b)) +
            ((std::uint32_t{1} << 31) >> b) ) |
            (std::uint32_t{1} << 16);
        unsigned int y = _bzhi_u32(all_32, b);
        std::cout
            << std::setw(3) << std::setfill('0') << b << " "
            << "x:0x"    << std::hex << std::setw(8) << std::setfill('0') << x
            << " orig:0x" << std::setw(8) << _pdep_u32(all_32, x)
            << " emu:0x"  << std::setw(8) << pdep32_emu(all_32, x)
            << " | y:0x"  << std::setw(8) << y
            << " orig:0x" << std::setw(8) << _pdep_u32(all_32, y)
            << " emu:0x"  << std::setw(8) << pdep32_emu(all_32, y)
            << std::dec << "\n";
        assert(_pdep_u32(all_32, x) == pdep32_emu(all_32, x));
        assert(_pdep_u32(all_32, y) == pdep32_emu(all_32, y));
    }

    std::cout << "-----------------------------------" << std::endl;
    for (unsigned int b = 0; b <= 64; b++) {
        std::uint64_t x = (((1ULL) << b) + ((1ULL << 63) >> b)) | (1ULL << 32) | (1ULL << 32) | (1ULL << 16) | (1ULL << 48);
        std::uint64_t y = _bzhi_u64(all_64, b);
        std::cout
            << std::setw(3) << std::setfill('0') << b << " "
            << "x:0x"    << std::hex << std::setw(16) << std::setfill('0') << x
            << " orig:0x" << std::setw(16) << _pext_u64(all_64, x)
            << " emu:0x"  << std::setw(16) << pext64_emu(all_64, x)
            << " | y:0x"  << std::setw(16) << y
            << " orig:0x" << std::setw(16) << _pext_u64(all_64, y)
            << " emu:0x"  << std::setw(16) << pext64_emu(all_64, y)
            << std::dec << "\n";
        assert(_pext_u64(all_64, x) == pext64_emu(all_64, x));
        assert(_pext_u64(all_64, y) == pext64_emu(all_64, y));
    }

    std::cout << "-----------------------------------" << std::endl;
    for (unsigned int b = 0; b <= 64; b++) {
        std::uint64_t x = (((1ULL) << b) + ((1ULL << 63) >> b)) | (1ULL << 32) | (1ULL << 16) | (1ULL << 48);
        std::uint64_t y = _bzhi_u64(all_64, b);
        std::cout
            << std::setw(3) << std::setfill('0') << b << " "
            << "x:0x"    << std::hex << std::setw(16) << std::setfill('0') << x
            << " orig:0x" << std::setw(16) << _pdep_u64(all_64, x)
            << " emu:0x"  << std::setw(16) << pdep64_emu(all_64, x)
            << " | y:0x"  << std::setw(16) << y
            << " orig:0x" << std::setw(16) << _pdep_u64(all_64, y)
            << " emu:0x"  << std::setw(16) << pdep64_emu(all_64, y)
            << std::dec << "\n";
        assert(_pdep_u64(all_64, x) == pdep64_emu(all_64, x));
        assert(_pdep_u64(all_64, y) == pdep64_emu(all_64, y));
    }
}

#endif // SYSTEM_X86_64_GNU_CLANG
