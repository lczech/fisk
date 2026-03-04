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
//     Bit extract via hardware PEXT
// =================================================================================================

#if defined(HAVE_BMI2)

/**
 * @brief Bit extract via hardware PEXT from them BMI2 instruction set.
 */
inline std::uint64_t bit_extract_pext(std::uint64_t x, std::uint64_t mask)
{
    // For speed, not using bmi2_enabled() for check here, and instead assume
    // that hardware availablility means we are allowed to call it.
    return _pext_u64(x, mask);
}

#endif

// =================================================================================================
//     Bit extract via simple portable bit-loop
// =================================================================================================

/**
 * @brief Bit extract via a simple bit loop over all bits set in the mask.
 */
inline std::uint64_t bit_extract_bitloop(std::uint64_t x, std::uint64_t mask) noexcept
{
    // This is the classic "extract selected bits and pack them densely" loop.
    // Complexity ~64 iterations, but cheap operations.

    std::uint64_t out = 0;
    std::uint64_t out_bit = 1;
    std::uint64_t zero = 0;
    std::uint64_t one  = 1;

    while (mask) {
        std::uint64_t lsb = mask & (~mask + 1); // mask & -mask, but unsigned-safe
        out |= (x & lsb) ? out_bit : zero;
        mask ^= lsb;
        out_bit <<= one;
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
inline std::uint64_t bit_extract_split32(std::uint64_t x, std::uint64_t mask) noexcept
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
    std::uint32_t m_lo = static_cast<std::uint32_t>(mask);
    std::uint32_t m_hi = static_cast<std::uint32_t>(mask >> 32);

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
inline std::uint64_t bit_extract_byte_table(std::uint64_t x, std::uint64_t mask) noexcept
{
    // Use the precomputed bytes to implement a fast pext.
    static const BitExtractByteTable byte_table;

    std::uint64_t out = 0;
    unsigned shift = 0;

    for (int i = 0; i < 8; ++i) {
        std::uint8_t mm = static_cast<std::uint8_t>(mask >> (8 * i));
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
    std::array<std::uint64_t, 33> shifts{};
};

/**
 * @brief Compute the Block Shift lookup tables for a given mask.
 */
inline BitExtractBlockTable bit_extract_block_table_preprocess( std::uint64_t mask )
{
    BitExtractBlockTable table;
    // table.masks.resize( 32, 0 );
    // table.shifts.resize( 32, 0 );

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

        // Add to the table.
        table.blocks[arr_idx] = block_mask;
        table.shifts[arr_idx] = shift;
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
    std::uint64_t res = 0;
    size_t i = 0;
    while(bt.blocks[i]) {
        res |= (x & bt.blocks[i]) >> bt.shifts[i];
        ++i;
    }
    return res;
}

/**
 * @brief Bit extract via Block Shift table, 2-fold unrolled.
 */
inline std::uint64_t bit_extract_block_table_unrolled2(
    std::uint64_t x, BitExtractBlockTable const& bt
) noexcept {
    // Same as above, but 2-fold unrolled. We might overshoot, if the number of blocks of
    // consecutive 1s is not divisible by the unrolling size, but that is fine.
    // In that case, we are masking with zeros in those masks, so nothing happens.
    std::uint64_t res = 0;
    size_t i = 0;
    while(bt.blocks[i]) {
        res |= (x & bt.blocks[i+0]) >> bt.shifts[i+0];
        res |= (x & bt.blocks[i+1]) >> bt.shifts[i+1];
        i += 2;
    }
    return res;
}

/**
 * @brief Bit extract via Block Shift table, 4-fold unrolled.
 */
inline std::uint64_t bit_extract_block_table_unrolled4(
    std::uint64_t x, BitExtractBlockTable const& bt
) noexcept {
    // Same as above, but 4-fold unrolled.
    std::uint64_t res = 0;
    size_t i = 0;
    while(bt.blocks[i]) {
        res |= (x & bt.blocks[i+0]) >> bt.shifts[i+0];
        res |= (x & bt.blocks[i+1]) >> bt.shifts[i+1];
        res |= (x & bt.blocks[i+2]) >> bt.shifts[i+2];
        res |= (x & bt.blocks[i+3]) >> bt.shifts[i+3];
        i += 4;
    }
    return res;
}

/**
 * @brief Bit extract via Block Shift table, 8-fold unrolled.
 */
inline std::uint64_t bit_extract_block_table_unrolled8(
    std::uint64_t x, BitExtractBlockTable const& bt
) noexcept {
    // Same as above, but 8-fold unrolled.
    std::uint64_t res = 0;
    size_t i = 0;
    while(bt.blocks[i]) {
        res |= (x & bt.blocks[i+0]) >> bt.shifts[i+0];
        res |= (x & bt.blocks[i+1]) >> bt.shifts[i+1];
        res |= (x & bt.blocks[i+2]) >> bt.shifts[i+2];
        res |= (x & bt.blocks[i+3]) >> bt.shifts[i+3];
        res |= (x & bt.blocks[i+4]) >> bt.shifts[i+4];
        res |= (x & bt.blocks[i+5]) >> bt.shifts[i+5];
        res |= (x & bt.blocks[i+6]) >> bt.shifts[i+6];
        res |= (x & bt.blocks[i+7]) >> bt.shifts[i+7];
        i += 8;
    }
    return res;
}

// =================================================================================================
//     Bit extract with extraction network
// =================================================================================================

/**
 * @brief Table and values for the bit extraction via extraction network.
 *
 * The network works by shifting each position by increasing powers of two, such that each bit
 * has the chance to be moved to wherever it is needed in the final result.
 * The table here contains the sets of which bits need to be shifted in each step.
 */
struct BitExtractNetworkTable
{
    // The original mask, as well as subsets for each power of two.
    std::uint64_t mask = 0;
    std::array<std::uint64_t, 6> sieves{};
};

/**
 * @brief Compute the `mv` table for a given @p mask for bit extraction via extraction network.
 */
inline BitExtractNetworkTable bit_extract_network_table_preprocess( std::uint64_t mask )
{
    BitExtractNetworkTable out{};
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
 * @brief Bit extract via extraction network.
 */
inline std::uint64_t bit_extract_network_table(
    std::uint64_t x,
    BitExtractNetworkTable const& nt
) noexcept {
    x &= nt.mask;
    auto step = [&]( std::size_t s, std::uint64_t mv_i )
    {
        std::uint64_t t = x & mv_i;
        x = (x ^ t) | (t >> s);
    };
    step(  1, nt.sieves[0] );
    step(  2, nt.sieves[1] );
    step(  4, nt.sieves[2] );
    step(  8, nt.sieves[3] );
    step( 16, nt.sieves[4] );
    step( 32, nt.sieves[5] );
    return x;
}
