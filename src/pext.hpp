#pragma once

#include <array>
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

#include "cpu_intrinsics.hpp"

// =================================================================================================
//     Hardware PEXT
// =================================================================================================

static inline std::uint64_t pext_hw_bmi2_u64(std::uint64_t x, std::uint64_t mask)
{
    // For speed, not using bmi2_enabled() for check here, and instead assume
    // that hardware availablility means we are allowed to call it.
    #if defined(HAVE_BMI2)
        return _pext_u64(x, mask);
    #else
        (void)x; (void)mask;
        return 0;
    #endif
}

// =================================================================================================
//     Simple software PEXT implementations
// =================================================================================================

// -----------------------------------------------------------------------------
//     Simple portable bit-loop implementation
// -----------------------------------------------------------------------------

static inline std::uint64_t pext_sw_bitloop_u64(std::uint64_t x, std::uint64_t mask)
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

// -----------------------------------------------------------------------------
//     Split into two 32-bit halves
// -----------------------------------------------------------------------------

static inline std::uint64_t pext_sw_split32_u64(std::uint64_t x, std::uint64_t mask)
{
    // Same as above, but split into two 32-bit halves.
    // Sometimes generates slightly better code depending on compiler/flags.

    auto pext32 = [](std::uint32_t xx, std::uint32_t mm) -> std::uint32_t
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

    std::uint32_t out_lo = pext32(x_lo, m_lo);

    // number of bits extracted from low half determines shift for high half
    #if defined(__GNUC__) || defined(__clang__)
        unsigned shift = static_cast<unsigned>(__builtin_popcount(m_lo));
    #else
        unsigned shift = 0;
        for (std::uint32_t t = m_lo; t; t &= (t - 1)) ++shift;
    #endif

    std::uint32_t out_hi = pext32(x_hi, m_hi);
    return static_cast<std::uint64_t>(out_lo) | (static_cast<std::uint64_t>(out_hi) << shift);
}

// -----------------------------------------------------------------------------
//     Byte-wise table implementation (8-bit chunks)
// -----------------------------------------------------------------------------

struct PextTable8
{
    // Precomputes, for each 8-bit mask m and 8-bit value x, the packed result.
    // Also uses popcount(m) to know how much to shift the next chunk.

    std::array<std::array<std::uint8_t, 256>, 256> table{};
    std::array<std::uint8_t, 256> popcnt{};

    PextTable8()
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

static inline std::uint64_t pext_sw_table8_u64(std::uint64_t x, std::uint64_t mask)
{
    // Use the precomputed bytes to implement a fast pext.
    static const PextTable8 pext_table;

    std::uint64_t out = 0;
    unsigned shift = 0;

    for (int i = 0; i < 8; ++i) {
        std::uint8_t mm = static_cast<std::uint8_t>(mask >> (8 * i));
        std::uint8_t xx = static_cast<std::uint8_t>(x >> (8 * i));
        std::uint64_t part = static_cast<std::uint64_t>(pext_table.table[mm][xx]);
        out |= (part << shift);
        shift += pext_table.popcnt[mm];
    }
    return out;
}

// -----------------------------------------------------------------------------
//     Preprocessing with blocks
// -----------------------------------------------------------------------------

struct PextBlockTable
{
    // One entry per run of consecutive 1-bits in the original mask.
    // First, a mask selecting that run at its original bit positions.
    // Second, right-shift value to move that run to its packed output position.
    // In the worst case, we have an interleaved pattern of 32 ones and zeros.
    // std::vector<std::uint64_t> masks;
    // std::vector<std::uint64_t> shifts;
    std::array<std::uint64_t, 32> masks{};
    std::array<std::uint64_t, 32> shifts{};
};

inline PextBlockTable pext_sw_block_table_preprocess_u64( std::uint64_t mask )
{
    PextBlockTable table;
    // table.masks.resize( 32, 0 );
    // table.shifts.resize( 32, 0 );

    // Helper: build a contiguous run mask of length len at bit position start.
    auto make_run_mask64 = [](unsigned start, unsigned len) -> std::uint64_t
    {
        if (len == 0) return 0;
        if (len >= 64) return ~std::uint64_t{0};
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
        if( arr_idx == 32 ) {
            throw std::runtime_error( "pext_sw_block_table_preprocess_u64 with arr_idx == 32" );
        }

        // Add to the table.
        table.masks[arr_idx]  = block_mask;
        table.shifts[arr_idx] = shift;
        ++arr_idx;
        out_pos += len;
    }

    return table;
}

// Apply blockwise-PEXT using the preprocessing above.
// Semantics match _pext_u64(x, mask) for the same mask.
static inline std::uint64_t pext_sw_block_table_u64(std::uint64_t x, PextBlockTable const& pb)
{
    std::uint64_t res = 0;
    size_t i = 0;
    while(pb.masks[i]) {
        res |= (x & pb.masks[i]) >> pb.shifts[i];
        ++i;
    }
    return res;
}
