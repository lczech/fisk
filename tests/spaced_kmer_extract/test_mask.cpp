#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/bit_extract/simd.hpp"
#include "fisk/simd/scalar.hpp"
#include "fisk/core/types.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"
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

TEST(SpacedKmerExtractMask, MakeMask)
{
    EXPECT_EQ(make_spaced_kmer_mask("1011").mask, std::uint64_t{0xCF});
    EXPECT_EQ(make_spaced_kmer_mask("1011").span_k, std::size_t{4});
    EXPECT_EQ(make_spaced_kmer_mask("1*1").mask, std::uint64_t{0x33});
    EXPECT_EQ(make_spaced_kmer_mask(std::string(32, '1')).mask, ~std::uint64_t{0});
    EXPECT_EQ(make_spaced_kmer_mask(std::string(32, '1')).span_k, std::size_t{32});

    // LSB masks are mirrored, with the first position of the span in the lowest bits.
    EXPECT_EQ(make_spaced_kmer_mask<Layout::kLSB>("1011").mask, std::uint64_t{0xF3});
    EXPECT_EQ(make_spaced_kmer_mask<Layout::kLSB>("1011").span_k, std::size_t{4});
    EXPECT_EQ(make_spaced_kmer_mask<Layout::kLSB>("1101").mask, std::uint64_t{0xCF});
    EXPECT_EQ(make_spaced_kmer_mask<Layout::kLSB>("1*1").mask, std::uint64_t{0x33});
    EXPECT_EQ(make_spaced_kmer_mask<Layout::kLSB>(std::string(32, '1')).mask, ~std::uint64_t{0});
    EXPECT_ANY_THROW(make_spaced_kmer_mask<Layout::kLSB>("010"));
    EXPECT_ANY_THROW(make_spaced_kmer_mask<Layout::kLSB>("1x1"));

    EXPECT_ANY_THROW(make_spaced_kmer_mask(""));
    EXPECT_ANY_THROW(make_spaced_kmer_mask(std::string(33, '1')));
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
        EXPECT_EQ(mask.mask, make_spaced_kmer_mask<L>(strings[i]).mask);
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
            auto const expected = bit_extract_bit_loop(x, mask);
            EXPECT_EQ(bit_extract_butterfly(x, butterfly_mask), expected);
            EXPECT_EQ(bit_extract_block_shift(x, block_shift_mask), expected);
            EXPECT_EQ(kernel.bit_extract(x), expected);
            EXPECT_EQ(bit_extract_butterfly(x, butterfly_masks[i]), expected);
            EXPECT_EQ(bit_extract_block_shift(x, block_shift_masks[i]), expected);
            EXPECT_EQ(kernels[i].bit_extract(x), expected);
        }
    }
}

TEST(SpacedKmerExtractMask, MaskFactories)
{
    check_mask_factories<Layout::kMSB>();
    check_mask_factories<Layout::kLSB>();
}

// The factories set the validity mask to the kept positions, in the order of the string, for both
// layouts, and the other representations keep it.
template <Layout L>
static void check_validity_mask()
{
    for (auto const& mask_string : valid_spaced_kmer_masks()) {
        auto const mask = make_spaced_kmer_mask<L>(mask_string);
        std::uint64_t expected_validity = 0;
        for (std::size_t i = 0; i < mask_string.size(); ++i) {
            if (mask_string[i] == '1') {
                expected_validity |= std::uint64_t{1} << i;
            }
        }
        EXPECT_EQ(mask.validity_mask, expected_validity);
        EXPECT_EQ(make_spaced_kmer_butterfly_mask(mask).validity_mask, expected_validity);
        EXPECT_EQ(
            make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdScalar>>(mask).validity_mask,
            expected_validity
        );
    }
}

TEST(SpacedKmerExtractMask, ValidityMask)
{
    check_validity_mask<Layout::kMSB>();
    check_validity_mask<Layout::kLSB>();
}

TEST(SpacedKmerExtractMask, DispatchMaskSets)
{
    // A vector of up to kMaxDispatchedMasks masks is dispatched as a span of its size, and one
    // more mask throws.
    static_assert(kMaxDispatchedMasks == FISK_MAX_DISPATCHED_MASKS);
    std::vector<std::string> strings(kMaxDispatchedMasks, "1101");
    auto masks = make_spaced_kmer_masks(strings);
    auto const weight = dispatch_static_span<kMaxDispatchedMasks>(masks, [](auto span) {
        EXPECT_EQ(decltype(span)::extent, kMaxDispatchedMasks);
        std::size_t total = 0;
        for (auto const& mask : span) {
            total += spaced_kmer_weight(mask);
        }
        return total;
    });
    EXPECT_EQ(weight, 3 * kMaxDispatchedMasks);

    strings.push_back("1101");
    masks = make_spaced_kmer_masks(strings);
    EXPECT_THROW(
        dispatch_static_span<kMaxDispatchedMasks>(masks, [](auto) {}), std::invalid_argument
    );
}
