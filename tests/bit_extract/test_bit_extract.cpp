#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "bit_extract/common.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers and Oracle
// =================================================================================================

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

TEST(BitExtract, BitLoop)
{
    check_against_oracle([](std::uint64_t x, std::uint64_t mask) {
        return bit_extract_bit_loop(x, BitExtractMask(mask));
    });
}

TEST(BitExtract, Split32)
{
    check_against_oracle([](std::uint64_t x, std::uint64_t mask) {
        return bit_extract_split32(x, BitExtractMask(mask));
    });
}

TEST(BitExtract, ByteLookup)
{
    check_against_oracle([](std::uint64_t x, std::uint64_t mask) {
        return bit_extract_byte_lookup(x, BitExtractMask(mask));
    });
}

// The tables must not be constructible from a raw mask, as that would skip the preprocessing.
static_assert(!std::is_aggregate_v<BitExtractBlockTable>);
static_assert(!std::is_aggregate_v<BitExtractButterflyTable>);
static_assert(!std::is_constructible_v<BitExtractBlockTable, std::uint64_t>);
static_assert(!std::is_constructible_v<BitExtractButterflyTable, std::uint64_t>);

TEST(BitExtract, BlockShift)
{
    check_against_oracle([](std::uint64_t x, std::uint64_t mask) {
        auto const table = make_bit_extract_block_table(mask);
        return bit_extract_block_shift(x, table);
    });
}

// bit_extract_block_shift_unrolled<UF>() is the same algorithm at different compile-time unroll
// factors; run every supported UF against the same oracle rather than treating them as separate
// tests, since they are not distinct algorithms.
template <std::size_t UF>
static void check_block_shift_unrolled()
{
    check_against_oracle([](std::uint64_t x, std::uint64_t mask) {
        auto const table = make_bit_extract_block_table(mask);
        return bit_extract_block_shift_unrolled<UF>(x, table);
    });
}

TEST(BitExtract, BlockShiftUnrolled)
{
    check_block_shift_unrolled<1>();
    check_block_shift_unrolled<2>();
    check_block_shift_unrolled<4>();
    check_block_shift_unrolled<8>();
    check_block_shift_unrolled<16>();
    check_block_shift_unrolled<32>();
}

TEST(BitExtract, Butterfly)
{
    check_against_oracle([](std::uint64_t x, std::uint64_t mask) {
        auto const table = make_bit_extract_butterfly_table(mask);
        return bit_extract_butterfly(x, table);
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
