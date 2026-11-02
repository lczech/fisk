#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/bit_extract/simd.hpp"
#include "fisk/simd/scalar.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_spaced.hpp"
#include "fisk/core/types.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"
#include "fisk/util/random.hpp"
#include "corpus.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Compile-Time Guarantees
// =================================================================================================

static_assert(decltype(make_spaced_kmer_mask("1"))::layout == Layout::kMSB);
static_assert(decltype(make_spaced_kmer_mask<Layout::kLSB>("1"))::layout == Layout::kLSB);
static_assert(std::is_same_v<
    decltype(make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdScalar>, Layout::kLSB>("1")),
    SpacedKmerMask<Layout::kLSB, BitExtractKernelButterfly<SimdScalar>>
>);

// =================================================================================================
//     Mask Factories
// =================================================================================================

// The bits of the first piece, and the span, of a mask, by value, as the test macros keep
// references to their arguments.
template <typename Mask>
static std::uint64_t bits(Mask const& mask)
{
    return mask.mask_pieces[0].mask;
}

template <typename Mask>
static std::size_t span(Mask const& mask)
{
    return mask.span_k;
}

TEST(SpacedKmerExtractMask, MakeMask)
{
    EXPECT_EQ(bits(make_spaced_kmer_mask("1011")), std::uint64_t{0xCF});
    EXPECT_EQ(span(make_spaced_kmer_mask("1011")), std::size_t{4});
    EXPECT_EQ(bits(make_spaced_kmer_mask("1*1")), std::uint64_t{0x33});
    EXPECT_EQ(bits(make_spaced_kmer_mask(std::string(32, '1'))), ~std::uint64_t{0});
    EXPECT_EQ(span(make_spaced_kmer_mask(std::string(32, '1'))), std::size_t{32});

    // LSB masks are mirrored, with the first position of the span in the lowest bits.
    EXPECT_EQ(bits(make_spaced_kmer_mask<Layout::kLSB>("1011")), std::uint64_t{0xF3});
    EXPECT_EQ(span(make_spaced_kmer_mask<Layout::kLSB>("1011")), std::size_t{4});
    EXPECT_EQ(bits(make_spaced_kmer_mask<Layout::kLSB>("1101")), std::uint64_t{0xCF});
    EXPECT_EQ(bits(make_spaced_kmer_mask<Layout::kLSB>("1*1")), std::uint64_t{0x33});
    EXPECT_EQ(bits(make_spaced_kmer_mask<Layout::kLSB>(std::string(32, '1'))), ~std::uint64_t{0});
    EXPECT_ANY_THROW(make_spaced_kmer_mask<Layout::kLSB>("010"));
    EXPECT_ANY_THROW(make_spaced_kmer_mask<Layout::kLSB>("1x1"));

    EXPECT_ANY_THROW(make_spaced_kmer_mask(""));
    EXPECT_ANY_THROW(make_spaced_kmer_mask(std::string(33, '1')));

    // Spans above 32 up to the maximum span, with at most 32 kept positions.
    std::string const span_64 = std::string("1").append(62, '0').append("1");
    EXPECT_EQ(span(make_spaced_kmer_mask(span_64)), std::size_t{64});
    EXPECT_ANY_THROW((make_spaced_kmer_mask<Layout::kMSB, 32>(span_64)));
    EXPECT_ANY_THROW(make_spaced_kmer_mask(std::string("1").append(63, '0').append("1")));
    EXPECT_EQ((span(make_spaced_kmer_mask<Layout::kMSB, 65>(span_64 + "1"))), std::size_t{65});
    EXPECT_ANY_THROW(make_spaced_kmer_mask("010"));
    EXPECT_ANY_THROW(make_spaced_kmer_mask("1010"));
    EXPECT_ANY_THROW(make_spaced_kmer_mask("1x1"));
}

// All representations of a mask carry the same Layout, span and bits, and the vector overloads
// create the same masks as the single ones.
template <Layout L>
static void check_mask_factories()
{
    std::vector<std::string> const strings = {"1011", "1*0011"};
    auto const masks = make_spaced_kmer_masks<L>(strings);
    auto const butterfly_masks = make_spaced_kmer_butterfly_masks(masks);
    auto const block_shift_masks = make_spaced_kmer_block_shift_masks(masks);
    auto const kernels = make_spaced_kmer_kernels<BitExtractKernelBlockShift<SimdScalar>>(masks);
    ASSERT_EQ(masks.size(), strings.size());

    for (std::size_t i = 0; i < strings.size(); ++i) {
        auto const& mask = masks[i];
        EXPECT_EQ(bits(mask), bits(make_spaced_kmer_mask<L>(strings[i])));
        EXPECT_EQ(mask.span_k, strings[i].size());

        auto const butterfly_mask = make_spaced_kmer_butterfly_mask<L>(strings[i]);
        auto const block_shift_mask = make_spaced_kmer_block_shift_mask<L>(strings[i]);
        auto const kernel
            = make_spaced_kmer_kernel<BitExtractKernelBlockShift<SimdScalar>, L>(strings[i])
        ;
        static_assert(decltype(butterfly_mask)::layout == L);
        static_assert(decltype(block_shift_mask)::layout == L);
        static_assert(decltype(kernel)::layout == L);
        static_assert(std::remove_cvref_t<decltype(butterfly_masks[i])>::layout == L);
        static_assert(std::remove_cvref_t<decltype(block_shift_masks[i])>::layout == L);
        static_assert(std::remove_cvref_t<decltype(kernels[i])>::layout == L);
        EXPECT_EQ(butterfly_mask.span_k, mask.span_k);
        EXPECT_EQ(block_shift_mask.span_k, mask.span_k);
        EXPECT_EQ(kernel.span_k, mask.span_k);
        EXPECT_EQ(butterfly_masks[i].span_k, mask.span_k);
        EXPECT_EQ(block_shift_masks[i].span_k, mask.span_k);
        EXPECT_EQ(kernels[i].span_k, mask.span_k);

        // The tables and kernels extract the same bits as the plain mask.
        for (std::uint64_t const x : {0x0ull, ~0x0ull, 0x0123456789ABCDEFull}) {
            auto const expected = bit_extract_bit_loop(x, mask.mask_pieces[0]);
            EXPECT_EQ(bit_extract_butterfly(x, butterfly_mask.mask_pieces[0]), expected);
            EXPECT_EQ(bit_extract_block_shift(x, block_shift_mask.mask_pieces[0]), expected);
            EXPECT_EQ(kernel.mask_pieces[0].bit_extract(x), expected);
            EXPECT_EQ(bit_extract_butterfly(x, butterfly_masks[i].mask_pieces[0]), expected);
            EXPECT_EQ(bit_extract_block_shift(x, block_shift_masks[i].mask_pieces[0]), expected);
            EXPECT_EQ(kernels[i].mask_pieces[0].bit_extract(x), expected);
        }
    }
}

TEST(SpacedKmerExtractMask, MaskFactories)
{
    check_mask_factories<Layout::kMSB>();
    check_mask_factories<Layout::kLSB>();
}

// The factories set the validity piece to the kept positions, in the order of the string, for
// both layouts, and the other representations keep it.
template <Layout L>
static void check_validity_mask()
{
    for (auto const& mask_string : valid_spaced_kmer_masks()) {
        auto const mask = make_spaced_kmer_mask<L>(mask_string);
        std::uint32_t expected_validity = 0;
        for (std::size_t i = 0; i < mask_string.size(); ++i) {
            if (mask_string[i] == '1') {
                expected_validity |= std::uint32_t{1} << i;
            }
        }
        auto const butterfly_mask = make_spaced_kmer_butterfly_mask(mask);
        auto const kernel = make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdScalar>>(mask);
        EXPECT_EQ(mask.validity_pieces[0], expected_validity);
        EXPECT_EQ(butterfly_mask.validity_pieces[0], expected_validity);
        EXPECT_EQ(kernel.validity_pieces[0], expected_validity);
    }
}

TEST(SpacedKmerExtractMask, ValidityMask)
{
    check_validity_mask<Layout::kMSB>();
    check_validity_mask<Layout::kLSB>();
}

// =================================================================================================
//     Pieces
// =================================================================================================

// Masks of span 1 to 128, around the boundaries of the pieces, with random kept positions, at most
// 32 of them.
static std::vector<std::string> const& wide_test_masks()
{
    static std::vector<std::string> const masks = [] {
        Splitmix64 rng(9009);
        std::vector<std::string> out;
        for (std::size_t span_k : {
            1u, 2u, 29u, 30u, 32u, 33u, 34u, 56u, 57u, 58u, 64u, 84u, 85u, 86u, 100u, 112u, 113u,
            128u
        }) {
            for (std::size_t rep = 0; rep < 4; ++rep) {
                std::string mask(span_k, '0');
                std::size_t const weight = 2 + rng.get_uint64() % 31;
                for (std::size_t i = 0; i < weight; ++i) {
                    mask[rng.get_uint64() % span_k] = '1';
                }
                mask.front() = '1';
                mask.back() = '1';
                out.push_back(mask);
            }
        }
        return out;
    }();
    return masks;
}

// The spaced k-mer of a window under a mask, from its pieces: the bits extracted from the window of
// each piece, as the extraction loops arrange it for its span, shifted by its mask shift. This is
// the same as the spaced k-mer that spaced_kmer_encode() builds from the kept positions.
template <Layout L>
static void check_pieces()
{
    Splitmix64 rng(1001);
    for (auto const& mask_string : wide_test_masks()) {
        auto const mask = make_spaced_kmer_mask<L, 128>(mask_string);
        auto const butterfly_mask = make_spaced_kmer_butterfly_mask(mask);
        std::size_t const span_k = mask_string.size();
        std::size_t const count = spaced_kmer_mask_piece_count(span_k);
        EXPECT_EQ(count, span_k <= 32 ? 1 : (span_k - 29 + 55) / 28);
        EXPECT_EQ(spaced_kmer_mask_tail_span(span_k), span_k - 28 * (count - 1));
        EXPECT_TRUE(is_valid_spaced_kmer_mask(mask));
        EXPECT_TRUE(is_valid_spaced_kmer_mask(butterfly_mask));
        EXPECT_EQ(spaced_kmer_mask_string(mask), mask_string);
        EXPECT_EQ(spaced_kmer_mask_string(butterfly_mask), mask_string);
        EXPECT_EQ(spaced_kmer_weight(mask), spaced_kmer_weight(mask_string));

        // A shift that does not match the weights, and a piece beyond the piece count that is not
        // empty, make the mask invalid.
        auto wrong_shift = mask;
        wrong_shift.mask_shifts[count - 1] ^= 2;
        EXPECT_FALSE(is_valid_spaced_kmer_mask(wrong_shift));
        if (count < mask.max_pieces) {
            auto extra_piece = mask;
            extra_piece.mask_pieces[count] = BitExtractMask(3);
            EXPECT_FALSE(is_valid_spaced_kmer_mask(extra_piece));
        }

        for (std::size_t rep = 0; rep < 8; ++rep) {
            std::string const window = random_sequence(rng, span_k, "ACGT");
            std::uint64_t value = 0;
            std::uint64_t butterfly_value = 0;
            for (std::size_t j = 0; j < count; ++j) {
                std::size_t const len = j + 1 == count ? spaced_kmer_mask_tail_span(span_k) : 28;
                auto const piece = kmer_encode<Kmer<Encoding::kACGT, L>>(
                    window.substr(28 * j, len)
                );
                value |= bit_extract_bit_loop(kmer_value(piece), mask.mask_pieces[j])
                    << mask.mask_shifts[j];
                butterfly_value |= bit_extract_butterfly(
                    kmer_value(piece), butterfly_mask.mask_pieces[j]
                ) << butterfly_mask.mask_shifts[j];
            }
            auto const expected = spaced_kmer_encode<Encoding::kACGT>(window, mask);
            EXPECT_EQ(value, kmer_value(expected));
            EXPECT_EQ(butterfly_value, kmer_value(expected));
        }
    }
}

TEST(SpacedKmerExtractMask, Pieces)
{
    check_pieces<Layout::kMSB>();
    check_pieces<Layout::kLSB>();
}

// The sizes of the masks, with the piece members ordered by their alignment.
TEST(SpacedKmerExtractMask, Sizes)
{
    EXPECT_EQ((sizeof(SpacedKmerMask<Layout::kMSB, BitExtractMask, 32>)), std::size_t{16});
    EXPECT_EQ((sizeof(SpacedKmerMask<Layout::kMSB, BitExtractMask, 64>)), std::size_t{48});
    EXPECT_EQ((SpacedKmerMask<Layout::kMSB, BitExtractMask, 32>::max_pieces), std::size_t{1});
    EXPECT_EQ((SpacedKmerMask<Layout::kMSB, BitExtractMask, 57>::max_pieces), std::size_t{2});
    EXPECT_EQ((SpacedKmerMask<Layout::kMSB, BitExtractMask, 58>::max_pieces), std::size_t{3});
}

TEST(SpacedKmerExtractMask, DispatchMaskSets)
{
    // A vector of up to kDefaultMaxMasks masks is dispatched as a span of its size, and one
    // more mask throws.
    static_assert(kDefaultMaxMasks == FISK_DEFAULT_MAX_MASKS);
    std::vector<std::string> strings(kDefaultMaxMasks, "1101");
    auto masks = make_spaced_kmer_masks(strings);
    auto const weight = dispatch_static_span<kDefaultMaxMasks>(masks, [](auto span) {
        EXPECT_EQ(decltype(span)::extent, kDefaultMaxMasks);
        std::size_t total = 0;
        for (auto const& mask : span) {
            total += spaced_kmer_weight(mask);
        }
        return total;
    });
    EXPECT_EQ(weight, 3 * kDefaultMaxMasks);

    strings.push_back("1101");
    masks = make_spaced_kmer_masks(strings);
    EXPECT_THROW(
        dispatch_static_span<kDefaultMaxMasks>(masks, [](auto) {}), std::invalid_argument
    );
}
