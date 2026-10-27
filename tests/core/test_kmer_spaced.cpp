#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/bit_extract/simd.hpp"
#include "fisk/core/cast.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_spaced.hpp"
#include "fisk/simd/scalar.hpp"
#include "fisk/core/types.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers
// =================================================================================================

// A mask with arbitrary bits and span, bypassing make_spaced_kmer_mask(), to test invalid masks.
static SpacedKmerMask<Layout::kMSB, BitExtractMask> raw_spaced_kmer_mask(
    std::uint64_t bits, std::size_t span_k
) {
    SpacedKmerMask<Layout::kMSB, BitExtractMask> mask;
    static_cast<BitExtractMask&>(mask) = BitExtractMask(bits);
    mask.span_k = span_k;
    return mask;
}

// =================================================================================================
//     Spaced K-mer Encoding
// =================================================================================================

// Encoding a window with a mask keeps exactly the bases at the mask's '1' positions, in order, and
// must agree with the oracle under every convention.
template <typename K>
static void check_spaced_encode(std::string const& mask)
{
    std::size_t const weight = spaced_kmer_weight(mask);
    for (auto const& window : random_seqs(mask.size(), 8, 525252 + mask.size())) {
        std::string kept;
        std::string span;
        for (std::size_t i = 0; i < mask.size(); ++i) {
            if (mask[i] == '1') {
                kept.push_back(window[i]);
            }
            span.push_back(mask[i] == '1' ? window[i] : '-');
        }
        auto const spaced = spaced_kmer_encode<K>(window, mask);
        EXPECT_EQ(spaced, oracle_kmer<K>(kept));
        EXPECT_EQ(kmer_decode(spaced, weight), kept);
        EXPECT_EQ(spaced_kmer_decode(spaced, mask), span);
    }
}

TEST(CoreKmerSpaced, EncodeWithMask)
{
    for (std::string const& mask : std::vector<std::string>{
        "1", "11", "101", "1*011", "110100111", mask_for_span(31), std::string(32, '1')
    }) {
        check_spaced_encode<SpacedKmerAcgtMsb>(mask);
        check_spaced_encode<SpacedKmerAcgtLsb>(mask);
        check_spaced_encode<SpacedKmerActgMsb>(mask);
        check_spaced_encode<SpacedKmerActgLsb>(mask);
    }
}

// The result type is named as for the contiguous kmer_encode(), and defaults to ACGT/MSB. Only
// spaced k-mers take a mask, for encoding and decoding.
template <typename K>
concept EncodableWithMask = requires(std::string_view window, std::string const& mask) {
    spaced_kmer_encode<K>(window, mask);
};

static_assert(std::is_same_v<decltype(spaced_kmer_encode("ACGT", "1011")), SpacedKmerAcgtMsb>);
static_assert(std::is_same_v<
    decltype(spaced_kmer_encode<SpacedKmerActgLsb>("A", "1")), SpacedKmerActgLsb
>);
static_assert(EncodableWithMask<SpacedKmerAcgtMsb>);
static_assert(!EncodableWithMask<KmerAcgtMsb>);

template <typename K>
concept DecodableWithMask = requires(K const& kmer, std::string const& mask) {
    spaced_kmer_decode(kmer, mask);
};

static_assert(DecodableWithMask<SpacedKmerActgLsb>);

// With a tagged mask, the Layout of the result is the one of the mask, as during extraction.
static_assert(std::is_same_v<
    decltype(spaced_kmer_encode("A", make_spaced_kmer_mask("1"))), SpacedKmerAcgtMsb
>);
static_assert(std::is_same_v<
    decltype(spaced_kmer_encode<Encoding::kACTG>("A", make_spaced_kmer_mask("1"))),
    SpacedKmerActgMsb
>);
static_assert(std::is_same_v<
    decltype(spaced_kmer_encode("A", make_spaced_kmer_mask<Layout::kLSB>("1"))),
    SpacedKmerAcgtLsb
>);
static_assert(!DecodableWithMask<KmerAcgtMsb>);

TEST(CoreKmerSpaced, EncodeWithMaskDefaults)
{
    // ACGT under 1011 keeps A, G, T: codes 0, 2, 3, earliest in the high bits.
    EXPECT_EQ(kmer_value(spaced_kmer_encode("ACGT", "1011")), std::uint64_t{0x0B});
    EXPECT_EQ(spaced_kmer_encode("acgt", "1011"), spaced_kmer_encode("ACGT", "1011"));
}

TEST(CoreKmerSpaced, DecodeWithMask)
{
    EXPECT_EQ(spaced_kmer_decode(spaced_kmer_encode("ACGTA", "10101"), "10101"), "A-G-A");
    EXPECT_EQ(spaced_kmer_decode(spaced_kmer_encode("acgta", "1*101"), "1*101"), "A-G-A");
    EXPECT_EQ(spaced_kmer_decode(spaced_kmer_encode("T", "1"), "1"), "T");

    // Tagged masks, in any representation, behave as their mask string.
    auto const kmer = spaced_kmer_encode("ACGTA", "10101");
    auto const mask = make_spaced_kmer_mask("1*101");
    auto const kernel = make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdScalar>>(mask);
    EXPECT_EQ(spaced_kmer_encode("ACGTA", mask), kmer);
    EXPECT_EQ(spaced_kmer_encode("ACGTA", make_spaced_kmer_block_table(mask)), kmer);
    EXPECT_EQ(spaced_kmer_encode("ACGTA", kernel), kmer);
    EXPECT_EQ(spaced_kmer_decode(kmer, mask), "A-G-A");
    EXPECT_EQ(spaced_kmer_decode(kmer, make_spaced_kmer_butterfly_table(mask)), "A-G-A");
    EXPECT_EQ(spaced_kmer_decode(kmer, kernel), "A-G-A");
    EXPECT_EQ(
        spaced_kmer_encode<Encoding::kACTG>("ACGTA", mask),
        spaced_kmer_encode<SpacedKmerActgMsb>("ACGTA", "10101")
    );

    // An LSB mask encodes to an LSB spaced k-mer, but shows the same span.
    auto const lsb_kmer = spaced_kmer_encode<SpacedKmerAcgtLsb>("ACGTT", "11001");
    auto const lsb_mask = make_spaced_kmer_mask<Layout::kLSB>("11001");
    auto const lsb_kernel
        = make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdScalar>>(lsb_mask)
    ;
    EXPECT_EQ(spaced_kmer_encode("ACGTT", lsb_mask), lsb_kmer);
    EXPECT_EQ(spaced_kmer_encode("ACGTT", lsb_kernel), lsb_kmer);
    EXPECT_EQ(spaced_kmer_decode(lsb_kmer, lsb_mask), "AC--T");
    EXPECT_EQ(spaced_kmer_decode(lsb_kmer, lsb_kernel), "AC--T");
    EXPECT_EQ(kmer_value(lsb_kmer), std::uint64_t{0x34});

    EXPECT_THROW((void) spaced_kmer_decode(kmer, ""), std::invalid_argument);
    EXPECT_THROW((void) spaced_kmer_decode(kmer, "01101"), std::invalid_argument);
    EXPECT_THROW((void) spaced_kmer_decode(kmer, "10x01"), std::invalid_argument);
}

// Invalid masks, mismatched lengths and non-nucleotides at kept positions throw. Positions the mask
// skips are not inspected, as during extraction.
TEST(CoreKmerSpaced, EncodeWithMaskInvalidInput)
{
    EXPECT_THROW((void) spaced_kmer_encode("ACG", "1011"), std::invalid_argument);
    EXPECT_THROW((void) spaced_kmer_encode("ACGTA", "1011"), std::invalid_argument);
    EXPECT_THROW((void) spaced_kmer_encode("", ""), std::invalid_argument);
    EXPECT_THROW((void) spaced_kmer_encode("ACGT", "0111"), std::invalid_argument);
    EXPECT_THROW((void) spaced_kmer_encode("ACGT", "1110"), std::invalid_argument);
    EXPECT_THROW((void) spaced_kmer_encode("ACGT", "1x11"), std::invalid_argument);
    EXPECT_THROW(
        (void) spaced_kmer_encode(std::string(33, 'A'), std::string(33, '1')), std::invalid_argument
    );

    for (char const c : all_invalid_bytes()) {
        std::string window = "ACGT";
        window[2] = c;
        EXPECT_THROW((void) spaced_kmer_encode(window, "1111"), std::invalid_argument);
        EXPECT_EQ(spaced_kmer_encode(window, "1101"), spaced_kmer_encode("ACAT", "1101"));
        EXPECT_EQ(spaced_kmer_encode(window, "11*1"), spaced_kmer_encode("ACAT", "1101"));
    }
}

// =================================================================================================
//     Mask Helpers
// =================================================================================================

TEST(CoreKmerSpaced, Weight)
{
    EXPECT_EQ(spaced_kmer_weight(std::string("1")), std::size_t{1});
    EXPECT_EQ(spaced_kmer_weight(std::string("1011")), std::size_t{3});
    EXPECT_EQ(spaced_kmer_weight(std::string("1*1")), std::size_t{2});
    EXPECT_EQ(spaced_kmer_weight(std::string(32, '1')), std::size_t{32});

    EXPECT_EQ(spaced_kmer_weight(std::uint64_t{0xCF}), std::size_t{3});
    EXPECT_EQ(spaced_kmer_weight(std::uint64_t{0x33}), std::size_t{2});
    EXPECT_EQ(spaced_kmer_weight(~std::uint64_t{0}), std::size_t{32});

    EXPECT_EQ(spaced_kmer_weight(make_spaced_kmer_mask("1011")), std::size_t{3});
    EXPECT_EQ(spaced_kmer_weight(make_spaced_kmer_butterfly_table("1011")), std::size_t{3});
    EXPECT_EQ(spaced_kmer_weight(make_spaced_kmer_block_table("1011")), std::size_t{3});
    EXPECT_EQ(
        spaced_kmer_weight(make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdScalar>>("1011")),
        std::size_t{3}
    );

    EXPECT_ANY_THROW(spaced_kmer_weight(std::string("")));
    EXPECT_ANY_THROW(spaced_kmer_weight(std::string("010")));
}

TEST(CoreKmerSpaced, MaskPositions)
{
    EXPECT_EQ(spaced_kmer_mask_positions("1"), (std::vector<std::size_t>{0}));
    EXPECT_EQ(spaced_kmer_mask_positions("10*11"), (std::vector<std::size_t>{0, 3, 4}));

    EXPECT_ANY_THROW(spaced_kmer_mask_positions(""));
    EXPECT_ANY_THROW(spaced_kmer_mask_positions(std::string(33, '1')));
    EXPECT_ANY_THROW(spaced_kmer_mask_positions("010"));
    EXPECT_ANY_THROW(spaced_kmer_mask_positions("1010"));
    EXPECT_ANY_THROW(spaced_kmer_mask_positions("1x1"));
}

// Valid masks are valid and print as their string in either layout.
template <Layout L>
static void check_mask_formatting()
{
    for (auto const& mask_string : valid_spaced_kmer_masks()) {
        auto const mask = make_spaced_kmer_mask<L>(mask_string);
        EXPECT_TRUE(is_valid_spaced_kmer_mask(mask));
        EXPECT_TRUE(is_valid_spaced_kmer_mask(mask.mask, mask_string.size()));

        std::string normalized = mask_string;
        std::replace(normalized.begin(), normalized.end(), '*', '0');
        EXPECT_EQ(spaced_kmer_mask_string(mask), normalized);
        EXPECT_EQ(spaced_kmer_mask_string(make_spaced_kmer_butterfly_table(mask)), normalized);
        EXPECT_EQ(
            spaced_kmer_mask_string(
                make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdScalar>>(mask)
            ),
            normalized
        );

        // A validity mask that does not match the kept positions makes the mask invalid.
        auto mismatched = mask;
        mismatched.validity_mask ^= std::uint64_t{1} << (mask_string.size() - 1);
        EXPECT_TRUE(!is_valid_spaced_kmer_mask(mismatched));
    }

    for (std::size_t span_k = 1; span_k <= 32; ++span_k) {
        std::string const mask_string = mask_for_span(span_k);
        auto const mask = make_spaced_kmer_mask<L>(mask_string);
        EXPECT_TRUE(is_valid_spaced_kmer_mask(mask));
        EXPECT_EQ(spaced_kmer_mask_string(mask), mask_string);
    }
}

TEST(CoreKmerSpaced, MaskValidationAndFormatting)
{
    check_mask_formatting<Layout::kMSB>();
    check_mask_formatting<Layout::kLSB>();

    EXPECT_TRUE(!is_valid_spaced_kmer_mask(std::uint64_t{0x3}, 2));
    EXPECT_TRUE(!is_valid_spaced_kmer_mask(std::uint64_t{0xD}, 2));
    EXPECT_TRUE(!is_valid_spaced_kmer_mask(std::uint64_t{0x3F}, 2));

    // A middle position that uses only one of its two bits, with both ends kept.
    EXPECT_TRUE(is_valid_spaced_kmer_mask(std::uint64_t{0x33}, 3));
    EXPECT_TRUE(!is_valid_spaced_kmer_mask(std::uint64_t{0x37}, 3));
    EXPECT_TRUE(!is_valid_spaced_kmer_mask(std::uint64_t{0x3B}, 3));

    // Spans outside of [1, 32] are invalid for any mask.
    EXPECT_TRUE(!is_valid_spaced_kmer_mask(std::uint64_t{0}, 0));
    EXPECT_TRUE(!is_valid_spaced_kmer_mask(std::uint64_t{0x3}, 0));
    EXPECT_TRUE(!is_valid_spaced_kmer_mask(~std::uint64_t{0}, 33));

    EXPECT_TRUE(!is_valid_spaced_kmer_mask(raw_spaced_kmer_mask(0x3, 2)));
    EXPECT_EQ(spaced_kmer_mask_string(raw_spaced_kmer_mask(0x3, 2)), "01");
    EXPECT_ANY_THROW(spaced_kmer_mask_string(raw_spaced_kmer_mask(0x1, 1)));
    EXPECT_ANY_THROW(spaced_kmer_mask_string(raw_spaced_kmer_mask(0x3F, 2)));
}
