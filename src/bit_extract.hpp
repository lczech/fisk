#pragma once

#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <limits>

#ifdef HAVE_BMI2
#include <immintrin.h>
#endif

#ifdef HAVE_CLMUL
#include <wmmintrin.h>
#endif

#include "sys_info.hpp"

// =================================================================================================
//     Bit extract mask
// =================================================================================================

/**
 * @brief Strongly typed bit extract mask.
 *
 * This is to ensure we have a uniform interface for all bit extraction algorithms,
 * and to avoid ambiguity in their usage, as bit extract otherwise takes two uints.
 */
struct BitExtractMask
{
    // Mask, accessible as a public member.
    std::uint64_t mask;

    // Constructors, default and taking the mask value.
    BitExtractMask() = default;
    constexpr explicit BitExtractMask( std::uint64_t m ) noexcept : mask{m} {}

    // Operator conversion
    constexpr operator std::uint64_t() const noexcept { return mask; }
};

// =================================================================================================
//     Bit extract via hardware PEXT
// =================================================================================================

#if defined(HAVE_BMI2)

/**
 * @brief Bit extract via hardware PEXT from the BMI2 instruction set.
 */
inline std::uint64_t bit_extract_pext(std::uint64_t x, BitExtractMask mask)
{
    // For speed, not using bmi2_enabled() for check here, and instead assume
    // that hardware availablility means we are allowed to call it.
    return _pext_u64(x, mask.mask);
}

#endif

// =================================================================================================
//     Bit extract via simple portable bit-loop
// =================================================================================================

/**
 * @brief Bit extract via a simple bit loop over all bits set in the mask.
 */
inline std::uint64_t bit_extract_bitloop(std::uint64_t x, BitExtractMask mask) noexcept
{
    // This is the classic "extract selected bits and pack them densely" loop.
    // Complexity ~64 iterations, but cheap operations.

    std::uint64_t out = 0;
    std::uint64_t bit = 1;
    std::uint64_t msk = mask.mask;

    while (msk) {
        std::uint64_t lsb = msk & (~msk + 1); // mask & -mask, but unsigned-safe
        out |= (x & lsb) ? bit : 0;
        msk ^= lsb;
        bit <<= 1;
    }
    return out;
}

// =================================================================================================
//     Bit extract via split into two 32-bit halves
// =================================================================================================

/**
 * @brief Bit extract by splitting the 64-bit word into two 32-bit parts, and running
 * the bit loop extraction separately.
 *
 * This might be a bit faster, but mostly just used out of curiosity. Not really that useful.
 */
inline std::uint64_t bit_extract_split32(std::uint64_t x, BitExtractMask mask) noexcept
{
    // Same as above, but split into two 32-bit halves.
    // Sometimes generates slightly better code depending on compiler/flags.

    auto bit_extract_bitloop32_ = [](std::uint32_t xx, std::uint32_t mm) -> std::uint32_t
    {
        std::uint32_t out = 0;
        std::uint32_t out_bit = 1;
        std::uint32_t zero = 0;
        std::uint32_t one  = 1;

        while (mm) {
            std::uint32_t lsb = mm & (~mm + 1u);
            out |= (xx & lsb) ? out_bit : zero;
            mm ^= lsb;
            out_bit <<= one;
        }
        return out;
    };

    std::uint32_t x_lo = static_cast<std::uint32_t>(x);
    std::uint32_t x_hi = static_cast<std::uint32_t>(x >> 32);
    std::uint32_t m_lo = static_cast<std::uint32_t>(mask.mask);
    std::uint32_t m_hi = static_cast<std::uint32_t>(mask.mask >> 32);

    std::uint32_t out_lo = bit_extract_bitloop32_(x_lo, m_lo);

    // number of bits extracted from low half determines shift for high half
    // #ifdef SYSTEM_X86_64_GNU_CLANG
    //     unsigned shift = static_cast<unsigned>(__builtin_popcount(m_lo));
    // #else
    //     unsigned shift = 0;
    //     for (std::uint32_t t = m_lo; t; t &= (t - 1)) ++shift;
    // #endif
    auto const shift = std::popcount(m_lo);

    std::uint32_t out_hi = bit_extract_bitloop32_(x_hi, m_hi);
    return static_cast<std::uint64_t>(out_lo) | (static_cast<std::uint64_t>(out_hi) << shift);
}

// =================================================================================================
//     Bit extract via byte-wise lookup table (8-bit chunks)
// =================================================================================================

/**
 * @brief Lookup tables for bit extraction via the Byte Table implementation.
 *
 * This table is generic for all masks, and only needs to be computed once.
 */
struct BitExtractByteTable
{
    // Precomputes, for each 8-bit mask m and 8-bit value x, the packed result.
    // Also uses popcount(m) to know how much to shift the next chunk.

    std::array<std::array<std::uint8_t, 256>, 256> table{};
    std::array<std::uint8_t, 256> popcnt{};

    BitExtractByteTable()
    {
        for (size_t m = 0; m < 256; ++m) {
            std::uint8_t c = 0;
            for (int b = 0; b < 8; ++b) c += (m >> b) & 1;
            popcnt[m] = c;

            for (size_t x = 0; x < 256; ++x) {
                std::uint8_t out = 0;
                std::uint8_t out_bit = 1;
                std::uint8_t mm = static_cast<std::uint8_t>(m);
                for (int b = 0; b < 8; ++b) {
                    std::uint8_t bit = static_cast<std::uint8_t>(1u << b);
                    if (mm & bit) {
                        if (x & bit) out |= out_bit;
                        out_bit <<= 1;
                    }
                }
                table[m][x] = out;
            }
        }
    }
};

/**
 * @brief Bit extract via Byte Table lookup.
 *
 * This implementation uses a precomputed table of each compbination of input byte and mask byte,
 * and loops over all bytes to extract them individually, then shifts them to the correct position
 * using a second, smaller, lookup table for each byte.
 */
inline std::uint64_t bit_extract_byte_table(std::uint64_t x, BitExtractMask mask) noexcept
{
    // Use the precomputed bytes to implement a fast pext.
    static const BitExtractByteTable byte_table;

    std::uint64_t out = 0;
    unsigned shift = 0;

    for (int i = 0; i < 8; ++i) {
        std::uint8_t mm = static_cast<std::uint8_t>(mask.mask >> (8 * i));
        std::uint8_t xx = static_cast<std::uint8_t>(x >> (8 * i));
        std::uint64_t part = static_cast<std::uint64_t>(byte_table.table[mm][xx]);
        out |= (part << shift);
        shift += byte_table.popcnt[mm];
    }
    return out;
}

// =================================================================================================
//     Bit extract with block table
// =================================================================================================

/**
 * @brief Lookup tables for bit extraction via the Block Shift implementation.
 *
 * These tables are specific for a given mask. If only few masks are used, as in our case here,
 * this is a reasonably fast implementation. With arbitrary masks in each bit extraction call,
 * such as in chess computing, this is not the right choice.
 */
struct BitExtractBlockTable
{
    // The original underlying bit mask.
    std::uint64_t mask{};

    // One entry per run of consecutive 1-bits in the original mask.
    // First, a mask selecting that run at its original bit positions.
    // Second, right-shift value to move that run to its packed output position.
    // In the worst case, we have an interleaved pattern of 32 ones and zeros.
    // We add one last entry to that, to make our loop unrolling a bit easier,
    // as we can then simply overshoot without invalid memory access.
    // The last entry needed for an actual mask (likely, with fewer blocks)
    // is indicated by the first mask that is 0. This way, we do not have to
    // add another variable here to keep track of the number of blocks.
    std::array<std::uint64_t, 33> blocks{};
    std::array<std::uint8_t,  33> shifts{};
};

/**
 * @brief Compute the Block Shift lookup tables for a given mask.
 */
inline BitExtractBlockTable bit_extract_block_table_preprocess( std::uint64_t mask )
{
    BitExtractBlockTable table{};
    table.mask = mask;

    // Helper: build a contiguous run mask of length len at bit position start.
    auto make_run_mask64 = [](unsigned start, unsigned len) -> std::uint64_t
    {
        if (len == 0) {
            return 0;
        }
        if (len >= 64) {
            return ~std::uint64_t{0};
        }
        return ((std::uint64_t{1} << len) - 1) << start;
    };

    unsigned out_pos = 0; // number of extracted bits assigned so far (packed output bit index)
    unsigned bit = 0;
    size_t arr_idx = 0;
    while (bit < 64) {
        // Skip zeros
        if (((mask >> bit) & 1ULL) == 0ULL) {
            ++bit;
            continue;
        }

        // Found start of a run of ones. Build a mask for that run.
        unsigned start = bit;
        while (bit < 64 && (((mask >> bit) & 1ULL) == 1ULL)) {
            ++bit;
        }
        unsigned end = bit - 1;
        unsigned len = end - start + 1;
        std::uint64_t block_mask = make_run_mask64(start, len);

        // In PEXT output, this run occupies [out_pos .. out_pos+len-1].
        // The bits are currently at [start .. start+len-1].
        // So shift right by (start - out_pos) to align them.
        // out_pos is always <= start.
        unsigned shift = start - out_pos;

        // Sanity check. Should not be possible to have more than 32 values.
        assert( arr_idx < 32 );
        assert( shift < 64 );

        // Add to the table.
        table.blocks[arr_idx] = block_mask;
        table.shifts[arr_idx] = static_cast<std::uint8_t>(shift);
        ++arr_idx;
        out_pos += len;
    }

    return table;
}

/**
 * @brief Bit extract via Block Shift table.
 *
 * This implementation uses a BitExtractBlockTable that is precomputed for the mask, and runs
 * as many iterations as there are blocks of consecutive 1s in the mask.
 */
inline std::uint64_t bit_extract_block_table(
    std::uint64_t x, BitExtractBlockTable const& bt
) noexcept {
    // Apply blockwise bit extraction using the preprocessing above.
    // Semantics match _pext_u64(x, mask) for the same mask.
    assert( bt.blocks[32] == 0 && bt.shifts[32] == 0 );
    std::uint64_t res = 0;
    size_t i = 0;
    while(bt.blocks[i]) {
        res |= (x & bt.blocks[i]) >> bt.shifts[i];
        ++i;
    }
    return res;
}

/**
 * @brief Helper for bit_extract_block_table_unrolled() to run the unrolled block table for one chunk.
 */
template<std::size_t... Is>
inline std::uint64_t bit_extract_block_table_chunk_(
    std::uint64_t x,
    BitExtractBlockTable const& bt,
    std::size_t i,
    std::index_sequence<Is...>
) noexcept
{
    std::uint64_t res = 0;
    ((res |= (x & bt.blocks[i + Is]) >> bt.shifts[i + Is]), ...);
    return res;
}

/**
 * @brief Bit extract via Block Shift table, with compile-time loop unrolling
 * according to a given unrolling factor `UF`.
 */
template<std::size_t UF = 1>
inline std::uint64_t bit_extract_block_table_unrolled(
    std::uint64_t x,
    BitExtractBlockTable const& bt
) noexcept
{
    static_assert(
        UF == 1 || UF == 2 || UF == 4 || UF == 8 || UF == 16 || UF == 32,
        "Supported unroll factors are 1, 2, 4, 8, 16, 32."
    );
    assert(bt.blocks[32] == 0 && bt.shifts[32] == 0);

    // Loop over chunks of the size of the unrolling factor,
    // until we reach the block sential (the first block that is 0).
    constexpr auto idx = std::make_index_sequence<UF>{};
    std::uint64_t res = 0;
    std::size_t i = 0;
    while( bt.blocks[i] ) {
        res |= bit_extract_block_table_chunk_(x, bt, i, idx);
        i += UF;
    }

    return res;
}

// =================================================================================================
//     Bit extract with extraction butterfly permutation network
// =================================================================================================

/**
 * @brief Table and values for the bit extraction via extraction butterfly permutation network.
 *
 * The butterfly network works by shifting each position by increasing powers of two, such that
 * each bit has the chance to be moved to wherever it is needed in the final result.
 * The table here contains the sets of which bits need to be shifted in each step.
 *
 * See https://stackoverflow.com/a/16752267 and https://stackoverflow.com/a/28283007 for the
 * inspiration for this implementation.
 */
struct BitExtractButterflyTable
{
    // The original mask, as well as subsets for each power of two.
    std::uint64_t mask = 0;
    std::array<std::uint64_t, 6> sieves{};
};

/**
 * @brief Compute the `mv` table for a given @p mask for bit extraction via butterfly network.
 */
inline BitExtractButterflyTable bit_extract_butterfly_table_preprocess( std::uint64_t mask )
{
    BitExtractButterflyTable out{};
    out.mask = mask;
    std::uint64_t m = mask;
    std::uint64_t mk = ~m << 1;

    for( size_t i = 0; i < 6; ++i ) {
        std::uint64_t mp = mk ^ (mk << 1);
        mp ^= (mp << 2);
        mp ^= (mp << 4);
        mp ^= (mp << 8);
        mp ^= (mp << 16);
        mp ^= (mp << 32);

        std::uint64_t mv = mp & m;
        out.sieves[i] = mv;

        const int s = (1 << i);
        m  = (m ^ mv) | (mv >> s);
        mk = mk & ~mp;
    }
    return out;
}

/**
 * @brief Bit extract via extraction butterfly network.
 */
inline std::uint64_t bit_extract_butterfly_table(
    std::uint64_t x,
    BitExtractButterflyTable const& bf
) noexcept {
    x &= bf.mask;
    auto step = [&]( std::size_t s, std::uint64_t mv_i )
    {
        std::uint64_t t = x & mv_i;
        x = (x ^ t) | (t >> s);
    };
    step(  1, bf.sieves[0] );
    step(  2, bf.sieves[1] );
    step(  4, bf.sieves[2] );
    step(  8, bf.sieves[3] );
    step( 16, bf.sieves[4] );
    step( 32, bf.sieves[5] );
    return x;
}
