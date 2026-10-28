#include <cctype>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/core/cast.hpp"
#include "fisk/core/char_encoder.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_spaced.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/ascii_rolling_scalar.hpp"
#include "fisk/spaced_kmer_extract/ascii_rolling_scalar.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"
#include "spaced_kmer_extract/common.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers
// =================================================================================================

// Checks one scalar bit extractor for every sequence in the corpus, through the overload for a set
// of masks, with the vector of masks as a span of fixed size, via dispatch_static_span(), and, for
// a single mask, also through the single-mask overload.
template <typename Enc, typename Mask, typename BitExtract>
static void check_scalar_variant(
    std::vector<std::string> const& mask_strings,
    std::vector<Mask> const& masks,
    Enc const& enc,
    BitExtract&& bit_extract
) {
    std::size_t const span_k = mask_strings.front().size();
    for (auto const& seq : sequence_corpus(span_k, 8)) {
        auto const expected = oracle_spaced_events<Enc::encoding, Mask::layout>(seq, mask_strings);
        auto const runs = collect_spaced_callback_forms<Mask::layout, false>(
            [&](auto const& callback) {
                dispatch_static_span<kMaxTestMasks>(masks, [&](auto mask_span) {
                    for_each_spaced_kmer_ascii_checked_rolling_scalar(
                        seq, mask_span, enc, bit_extract, callback
                    );
                });
            }
        );
        for (auto const& got : runs) {
            check_spaced_events(got, expected);
        }

        if (masks.size() == 1) {
            auto const single_runs = collect_spaced_callback_forms<Mask::layout, true>(
                [&](auto const& callback) {
                    for_each_spaced_kmer_ascii_checked_rolling_scalar(
                        seq, masks.front(), enc, bit_extract, callback
                    );
                }
            );
            for (auto const& got : single_runs) {
                check_spaced_events(got, expected);
            }
        }
    }
}

template <Layout L, typename Enc>
static void check_all_scalar_variants(
    std::vector<std::string> const& mask_strings, Enc const& enc
) {
    auto const direct_masks = make_spaced_kmer_masks<L>(mask_strings);
    auto const block_masks = make_spaced_kmer_block_shift_masks(direct_masks);
    auto const butterfly_masks = make_spaced_kmer_butterfly_masks(direct_masks);

    check_scalar_variant(
        mask_strings, direct_masks, enc,
        [](std::uint64_t x, BitExtractMask const& mask) { return bit_extract_bit_loop(x, mask); }
    );
    check_scalar_variant(
        mask_strings, direct_masks, enc,
        [](std::uint64_t x, BitExtractMask const& mask) { return bit_extract_byte_lookup(x, mask); }
    );
    check_scalar_variant(
        mask_strings, block_masks, enc,
        [](std::uint64_t x, BitExtractBlockShiftMask const& mask) {
            return bit_extract_block_shift(x, mask);
        }
    );
    check_scalar_variant(
        mask_strings, block_masks, enc,
        [](std::uint64_t x, BitExtractBlockShiftMask const& mask) {
            return bit_extract_block_shift_unrolled<2>(x, mask);
        }
    );
    check_scalar_variant(
        mask_strings, block_masks, enc,
        [](std::uint64_t x, BitExtractBlockShiftMask const& mask) {
            return bit_extract_block_shift_unrolled<4>(x, mask);
        }
    );
    check_scalar_variant(
        mask_strings, block_masks, enc,
        [](std::uint64_t x, BitExtractBlockShiftMask const& mask) {
            return bit_extract_block_shift_unrolled<8>(x, mask);
        }
    );
    check_scalar_variant(
        mask_strings, butterfly_masks, enc,
        [](std::uint64_t x, BitExtractButterflyMask const& mask) {
            return bit_extract_butterfly(x, mask);
        }
    );

    #if defined(FISK_HAS_BMI2)
    check_scalar_variant(
        mask_strings, direct_masks, enc,
        [](std::uint64_t x, BitExtractMask const& mask) { return bit_extract_pext(x, mask); }
    );
    #endif
}

template <Layout L, typename Enc>
static void check_scalar_masks(Enc const& enc)
{
    for (std::size_t span_k = 1; span_k <= 32; ++span_k) {
        check_all_scalar_variants<L>(std::vector<std::string>{mask_for_span(span_k)}, enc);
    }
    check_all_scalar_variants<L>({"1"}, enc);
    check_all_scalar_variants<L>({"10101", "11111", "10001"}, enc);
    check_all_scalar_variants<L>({"11011", "11101", "10111"}, enc);
    check_all_scalar_variants<L>({"10000000000000000000000000000001"}, enc);
}

// =================================================================================================
//     Compile-Time Guarantees
// =================================================================================================

// Spaced k-mer extraction only accepts masks that are tagged with their Layout.
template <typename Mask>
concept CanExtractWith = requires(std::string_view seq, Mask const& mask) {
    for_each_spaced_kmer_ascii_checked_rolling_scalar(
        seq, mask, CharEncoderTable<Encoding::kACGT>{}, bit_extract_bit_loop,
        [](std::size_t, auto const&) {}
    );
};

static_assert(CanExtractWith<SpacedKmerMask<Layout::kMSB, BitExtractMask>>);
static_assert(!CanExtractWith<BitExtractMask>);

// =================================================================================================
//     Emitted K-mers
// =================================================================================================

// Emitted values are typed k-mers under the encoder's encoding and the mask's layout. Decoding one
// with the weight of its mask must give exactly the bases kept by that mask, with no bits above the
// weight, and it must equal spaced_kmer_encode() of the same window and mask, given as string or as
// the tagged mask used for extraction. Decoding it with its mask gives the window, with gaps.
template <Layout L>
static void check_emitted_kmers()
{
    std::vector<std::string> const mask_strings = {"1011", "1001"};
    std::size_t const span_k = 4;
    auto const masks = make_spaced_kmer_masks<L>(mask_strings);

    std::size_t emitted = 0;
    for (auto const& seq : sequence_corpus(span_k, 8)) {
        for_each_spaced_kmer_ascii_checked_rolling_scalar(
            std::string_view(seq), std::span<SpacedKmerMask<L, BitExtractMask> const, 2>(masks),
            CharEncoderTable<Encoding::kACGT>{},
            bit_extract_bit_loop,
            [&](std::size_t pos, std::size_t mask_idx, auto const& kmer) {
                using K = SpacedKmer<Encoding::kACGT, L>;
                static_assert(std::is_same_v<std::remove_cvref_t<decltype(kmer)>, K>);
                auto const window = seq.substr(pos, span_k);
                EXPECT_EQ(kmer, spaced_kmer_encode<K>(window, mask_strings[mask_idx]));
                EXPECT_EQ(kmer, spaced_kmer_encode(window, masks[mask_idx]));

                auto const weight = spaced_kmer_weight(mask_strings[mask_idx]);
                EXPECT_EQ(kmer_value(kmer) >> (2 * weight), std::uint64_t{0});

                std::string expected;
                std::string expected_span;
                for (std::size_t p = 0; p < span_k; ++p) {
                    char const base = static_cast<char>(
                        std::toupper(static_cast<unsigned char>(seq[pos + p]))
                    );
                    if (mask_strings[mask_idx][p] == '1') {
                        expected += base;
                    }
                    expected_span += (mask_strings[mask_idx][p] == '1') ? base : '-';
                }
                EXPECT_EQ(kmer_decode(kmer, weight), expected);
                EXPECT_EQ(spaced_kmer_decode(kmer, mask_strings[mask_idx]), expected_span);
                EXPECT_EQ(spaced_kmer_decode(kmer, masks[mask_idx]), expected_span);
                ++emitted;
            }
        );
    }
    EXPECT_TRUE(emitted > 0);
}

TEST(SpacedKmerExtractAsciiRollingScalar, EmittedKmersAreTypedAndDecodeToKeptBases)
{
    check_emitted_kmers<Layout::kMSB>();
    check_emitted_kmers<Layout::kLSB>();
}

// The same extraction in both layouts emits at the same positions, and each LSB spaced k-mer is
// the layout conversion of the MSB one. The mask bits of both layouts only agree for palindromic
// masks, while the values differ for any mask of weight above 1.
TEST(SpacedKmerExtractAsciiRollingScalar, LayoutsAgreeViaConversion)
{
    std::vector<std::string> const mask_strings = {"11101", "10101", "11011", "10001", "1100111"};
    for (auto const& mask_string : mask_strings) {
        auto const msb_mask = make_spaced_kmer_mask<Layout::kMSB>(mask_string);
        auto const lsb_mask = make_spaced_kmer_mask<Layout::kLSB>(mask_string);
        std::size_t const weight = spaced_kmer_weight(mask_string);
        std::string const reversed(mask_string.rbegin(), mask_string.rend());
        EXPECT_EQ(msb_mask.mask == lsb_mask.mask, reversed == mask_string);

        bool any_differ = false;
        for (auto const& seq : sequence_corpus(mask_string.size(), 8)) {
            std::vector<std::pair<std::size_t, SpacedKmerAcgtMsb>> msb;
            std::vector<std::pair<std::size_t, SpacedKmerAcgtLsb>> lsb;
            for_each_spaced_kmer_ascii_checked_rolling_scalar(
                std::string_view(seq), msb_mask, CharEncoderTable<Encoding::kACGT>{},
                bit_extract_bit_loop,
                [&](std::size_t pos, auto const& kmer) { msb.emplace_back(pos, kmer); }
            );
            for_each_spaced_kmer_ascii_checked_rolling_scalar(
                std::string_view(seq), lsb_mask, CharEncoderTable<Encoding::kACGT>{},
                bit_extract_bit_loop,
                [&](std::size_t pos, auto const& kmer) { lsb.emplace_back(pos, kmer); }
            );

            ASSERT_EQ(msb.size(), lsb.size());
            for (std::size_t i = 0; i < msb.size(); ++i) {
                EXPECT_EQ(msb[i].first, lsb[i].first);
                EXPECT_EQ(kmer_convert<Layout::kLSB>(msb[i].second, weight), lsb[i].second);
                any_differ |= (kmer_value(msb[i].second) != kmer_value(lsb[i].second));
            }
        }
        EXPECT_TRUE(any_differ);
    }
}

// A mask that keeps every position is the contiguous k-mer, so the emitted spaced k-mers must match
// the contiguous extraction exactly, in order.
TEST(SpacedKmerExtractAsciiRollingScalar, AllOnesMaskMatchesContiguousKmers)
{
    for (std::size_t const k : std::vector<std::size_t>{1, 4, 12, 31, 32}) {
        auto const mask = make_spaced_kmer_mask(std::string(k, '1'));
        for (auto const& seq : sequence_corpus(k, 4)) {
            std::vector<std::uint64_t> spaced_values;
            for_each_spaced_kmer_ascii_checked_rolling_scalar(
                std::string_view(seq), mask, CharEncoderTable<Encoding::kACGT>{},
                bit_extract_bit_loop,
                [&](std::size_t /*pos*/, std::size_t /*mask_idx*/, auto const& kmer) {
                    spaced_values.push_back(kmer_value(kmer));
                }
            );

            std::vector<std::uint64_t> contiguous_values;
            for_each_kmer_ascii_checked_rolling_scalar(
                std::string_view(seq), k, CharEncoderTable<Encoding::kACGT>{},
                [&](auto const& kmer) { contiguous_values.push_back(kmer_value(kmer)); }
            );
            EXPECT_EQ(spaced_values, contiguous_values);
        }
    }
}

// =================================================================================================
//     Tests
// =================================================================================================

TEST(SpacedKmerExtractAsciiRollingScalar, Acgt)
{
    check_scalar_masks<Layout::kMSB>(CharEncoderTable<Encoding::kACGT>{});
    check_scalar_masks<Layout::kLSB>(CharEncoderTable<Encoding::kACGT>{});
}

TEST(SpacedKmerExtractAsciiRollingScalar, Actg)
{
    check_scalar_masks<Layout::kMSB>(CharEncoderTable<Encoding::kACTG>{});
    check_scalar_masks<Layout::kLSB>(CharEncoderTable<Encoding::kACTG>{});
}

// The span is set by make_spaced_kmer_mask(), so an invalid one is only possible by setting it by
// hand, and still has to be rejected.
TEST(SpacedKmerExtractAsciiRollingScalar, InvalidSpan)
{
    auto const extract = [](std::uint64_t x, BitExtractMask const& m) {
        return bit_extract_bit_loop(x, m);
    };
    check_invalid_k_throws(32, [&](std::string const& seq, std::size_t k) {
        auto mask = make_spaced_kmer_mask("1");
        mask.span_k = k;
        for_each_spaced_kmer_ascii_checked_rolling_scalar(
            seq, mask, CharEncoderTable<Encoding::kACGT>{}, extract,
            [](std::size_t, auto const&) {}
        );
    });
}

// All masks of one extraction share one rolling window, and hence need the same span.
TEST(SpacedKmerExtractAsciiRollingScalar, MixedSpans)
{
    std::string const seq = "ACGTACGTACGT";
    auto const callback = [](std::size_t, std::size_t, auto const&) {};

    auto const masks = make_spaced_kmer_masks({"101", "10001"});
    EXPECT_THROW(
        dispatch_static_span<kMaxTestMasks>(masks, [&](auto mask_span) {
            for_each_spaced_kmer_ascii_checked_rolling_scalar(
                seq, mask_span, CharEncoderTable<Encoding::kACGT>{}, bit_extract_bit_loop, callback
            );
        }),
        std::invalid_argument
    );
}

// An empty set of masks has no span of fixed size, so that its dispatch throws.
TEST(SpacedKmerExtractAsciiRollingScalar, EmptyMasks)
{
    std::vector<SpacedKmerMask<Layout::kMSB, BitExtractMask>> const masks;
    std::string const seq = "ACGTACGTACGT";
    EXPECT_THROW(
        dispatch_static_span<kMaxTestMasks>(masks, [&](auto mask_span) {
            for_each_spaced_kmer_ascii_checked_rolling_scalar(
                seq, mask_span, CharEncoderTable<Encoding::kACGT>{}, bit_extract_bit_loop,
                [](std::size_t, std::size_t, auto const&) {}
            );
        }),
        std::invalid_argument
    );
}
