#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/core/random.hpp"
#include "testing.hpp"

// =================================================================================================
//     Helpers and Oracle
// =================================================================================================

// Ground truth for bit extraction (PEXT semantics), written independently:
// Walk the mask from bit 0 up, and for every set mask bit, append the corresponding bit of `x`
// to the output. Every algorithm is checked against this for the same (x, mask).
static std::uint64_t bit_extract_oracle(std::uint64_t x, std::uint64_t mask)
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

static std::vector<std::uint64_t> const& edge_case_masks()
{
    static std::vector<std::uint64_t> const masks = {
        0x0000000000000000ULL,
        0xFFFFFFFFFFFFFFFFULL,
        0x0000000000000001ULL, // single bit, position 0
        0x0000000080000000ULL, // single bit, position 31
        0x8000000000000000ULL, // single bit, position 63
        0x5555555555555555ULL, // alternating bits: 32 runs of length 1, the block table's max
        0x00000000FFFF0000ULL, // one contiguous run
        0xF0F0F0F0F0F0F0F0ULL, // several contiguous runs of length 4
    };
    return masks;
}

static std::vector<std::uint64_t> const& random_masks()
{
    static std::vector<std::uint64_t> const masks = [] {
        std::vector<std::uint64_t> m;
        Splitmix64 rng(1001);
        for (int i = 0; i < 50; ++i) {
            m.push_back(rng.get_uint64());
        }
        return m;
    }();
    return masks;
}

static std::vector<std::uint64_t> const& random_values()
{
    static std::vector<std::uint64_t> const values = [] {
        std::vector<std::uint64_t> v;
        Splitmix64 rng(2002);
        for (int i = 0; i < 100; ++i) {
            v.push_back(rng.get_uint64());
        }
        return v;
    }();
    return values;
}

// Run `func(value, mask)` against bit_extract_oracle() for the full shared mask x value set.
template <typename Func>
static void check_against_oracle(Func&& func)
{
    std::vector<std::uint64_t> masks = edge_case_masks();
    for (auto const m : random_masks()) {
        masks.push_back(m);
    }

    std::vector<std::uint64_t> values = {
        0x0000000000000000ULL,
        0xFFFFFFFFFFFFFFFFULL,
    };
    for (auto const v : random_values()) {
        values.push_back(v);
    }

    for (auto const mask : masks) {
        for (auto const value : values) {
            EXPECT_EQ(func(value, mask), bit_extract_oracle(value, mask));
        }
    }
}

// =================================================================================================
//     Bit Extraction Tests
// =================================================================================================

TEST(BitExtract, Bitloop)
{
    check_against_oracle([](std::uint64_t x, std::uint64_t mask) {
        return bit_extract_bitloop(x, BitExtractMask(mask));
    });
}

TEST(BitExtract, Split32)
{
    check_against_oracle([](std::uint64_t x, std::uint64_t mask) {
        return bit_extract_split32(x, BitExtractMask(mask));
    });
}

TEST(BitExtract, ByteTable)
{
    check_against_oracle([](std::uint64_t x, std::uint64_t mask) {
        return bit_extract_byte_table(x, BitExtractMask(mask));
    });
}

TEST(BitExtract, BlockTable)
{
    check_against_oracle([](std::uint64_t x, std::uint64_t mask) {
        auto const table = bit_extract_block_table_preprocess(mask);
        return bit_extract_block_table(x, table);
    });
}

// bit_extract_block_table_unrolled<UF>() is the same algorithm at different compile-time unroll
// factors; run every supported UF against the same oracle rather than treating them as separate
// tests, since they are not distinct algorithms.
template <std::size_t UF>
static void check_block_table_unrolled()
{
    check_against_oracle([](std::uint64_t x, std::uint64_t mask) {
        auto const table = bit_extract_block_table_preprocess(mask);
        return bit_extract_block_table_unrolled<UF>(x, table);
    });
}

TEST(BitExtract, BlockTableUnrolled)
{
    check_block_table_unrolled<1>();
    check_block_table_unrolled<2>();
    check_block_table_unrolled<4>();
    check_block_table_unrolled<8>();
    check_block_table_unrolled<16>();
    check_block_table_unrolled<32>();
}

TEST(BitExtract, ButterflyTable)
{
    check_against_oracle([](std::uint64_t x, std::uint64_t mask) {
        auto const table = bit_extract_butterfly_table_preprocess(mask);
        return bit_extract_butterfly_table(x, table);
    });
}

#ifdef FISK_HAS_BMI2
TEST(BitExtract, Pext)
{
    check_against_oracle([](std::uint64_t x, std::uint64_t mask) {
        return bit_extract_pext(x, BitExtractMask(mask));
    });
}
#endif
