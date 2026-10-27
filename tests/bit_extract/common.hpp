#pragma once

#include <cstdint>
#include <vector>

#include "fisk/util/random.hpp"

// =================================================================================================
//     Helpers and Oracle
// =================================================================================================

// Ground truth for bit extraction (PEXT semantics), written independently:
// Walk the mask from bit 0 up, and for every set mask bit, append the corresponding bit of `x`
// to the output. Every algorithm is checked against this for the same (x, mask).
inline std::uint64_t bit_extract_oracle(std::uint64_t x, std::uint64_t mask)
{
    std::uint64_t out_val = 0;
    std::uint64_t out_pos = 0;
    for (unsigned bit = 0; bit < 64; ++bit) {
        std::uint64_t const bit_mask = std::uint64_t{1} << bit;
        if (mask & bit_mask) {
            if (x & bit_mask) {
                out_val |= (std::uint64_t{1} << out_pos);
            }
            ++out_pos;
        }
    }
    return out_val;
}

inline std::vector<std::uint64_t> const& edge_case_masks()
{
    static std::vector<std::uint64_t> const masks = {
        0x0000000000000000ULL,
        0xFFFFFFFFFFFFFFFFULL,
        0x0000000000000001ULL, // single bit, position 0
        0x0000000080000000ULL, // single bit, position 31
        0x8000000000000000ULL, // single bit, position 63
        0x5555555555555555ULL, // alternating bits: 32 runs of length 1, the most for block shift
        0x00000000FFFF0000ULL, // one contiguous run
        0xF0F0F0F0F0F0F0F0ULL, // several contiguous runs of length 4
    };
    return masks;
}

inline std::vector<std::uint64_t> const& random_masks()
{
    static std::vector<std::uint64_t> const masks = [] {
        std::vector<std::uint64_t> m;
        fisk::Splitmix64 rng(1001);
        for (int i = 0; i < 50; ++i) {
            m.push_back(rng.get_uint64());
        }
        return m;
    }();
    return masks;
}

inline std::vector<std::uint64_t> const& random_values()
{
    static std::vector<std::uint64_t> const values = [] {
        std::vector<std::uint64_t> v;
        fisk::Splitmix64 rng(2002);
        for (int i = 0; i < 100; ++i) {
            v.push_back(rng.get_uint64());
        }
        return v;
    }();
    return values;
}
