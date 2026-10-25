#include <cstdint>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/bit_extract/selector.hpp"
#include "bit_extract/common.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Selector Tests
// =================================================================================================

TEST(BitExtractSelector, RunBitExtractMode)
{
    // run_bit_extract_mode() dispatches a BitExtractMode to the corresponding already-tested
    // algorithm above; the point here is dispatch correctness (does each mode route to the right
    // implementation), not re-proving algorithm correctness.

    std::vector<BitExtractMode> modes = {
        BitExtractMode::kBlockTable,
        BitExtractMode::kBlockTableUnrolled2,
        BitExtractMode::kBlockTableUnrolled4,
        BitExtractMode::kBlockTableUnrolled8,
        BitExtractMode::kButterflyTable,
    };
#ifdef FISK_HAS_BMI2
    modes.push_back(BitExtractMode::kPext);
#endif

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

    for (auto const mode : modes) {
        for (auto const mask : masks) {
            run_bit_extract_mode(mode, BitExtractMask(mask), [&](auto&& extractor) {
                for (auto const value : values) {
                    EXPECT_EQ(extractor(value), bit_extract_oracle(value, mask));
                }
            });
        }
    }
}

TEST(BitExtractSelector, SmokeTest)
{
    // bit_extract_selector() picks whichever candidate is fastest right here right now; there
    // is no "correct" mode to assert, and it already self-checks candidate agreement internally
    // (throws on mismatch). This is a smoke test only: it completes without throwing and returns a
    // value within BitExtractMode's valid range, for a couple of masks.

    std::vector<std::uint64_t> const masks = {
        0x5555555555555555ULL, // alternating bits, maximal run count
        random_masks().front(),
    };

    for (auto const mask : masks) {
        auto const mode = bit_extract_selector(mask);
        bool const is_known_mode =
            mode == BitExtractMode::kPext ||
            mode == BitExtractMode::kBlockTable ||
            mode == BitExtractMode::kBlockTableUnrolled2 ||
            mode == BitExtractMode::kBlockTableUnrolled4 ||
            mode == BitExtractMode::kBlockTableUnrolled8 ||
            mode == BitExtractMode::kButterflyTable;
        EXPECT_TRUE(is_known_mode);
    }
}
