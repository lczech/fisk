#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/core/types.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"
#include "fisk/spaced_kmer_extract/packed_bytewise_scalar.hpp"
#include "spaced_kmer_extract/common.hpp"
#include "spaced_kmer_extract/packed_scalar_common.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Extraction
// =================================================================================================

// Checks the checked extraction against the oracle, which skips the spaced k-mers with an invalid
// character at a position that their mask keeps, and the unchecked one on the same sequence, which
// ignores the validity bits. The bit extraction does not depend on the validity, so that two
// extractors suffice here.
template <Encoding E, Layout L>
static void check_checked_extraction()
{
    WordEncoderButterfly<E, L> const encoder;
    for (auto const& mask_strings : test_mask_sets()) {
        auto const direct_masks = make_spaced_kmer_masks<L, kNarrowMaxSpan>(mask_strings);
        auto const butterfly_masks = make_spaced_kmer_butterfly_masks(direct_masks);

        for (auto const& seq : packed_checked_test_sequences()) {
            auto const packed = pack_sequence_checked(seq, encoder);
            auto const expected = oracle_spaced_events<E, L>(seq, mask_strings);

            auto const check = [&](auto const& masks, auto const& bit_ext) {
                check_callback_forms<L>(
                    masks, expected, [&](auto const& mask_or_masks, auto const& callback) {
                        for_each_spaced_kmer_packed_checked_bytewise_scalar(
                            packed, mask_or_masks, bit_ext, callback
                        );
                    }
                );
            };
            check(direct_masks, [](std::uint64_t x, BitExtractMask const& mask) {
                return bit_extract_bit_loop(x, mask);
            });
            check(butterfly_masks, [](std::uint64_t x, BitExtractButterflyMask const& mask) {
                return bit_extract_butterfly(x, mask);
            });

            // Unchecked, all positions, whose number the oracle gives for a sequence without
            // invalid characters.
            std::size_t calls = 0;
            dispatch_static_span<kMaxTestMasks>(direct_masks, [&](auto mask_span) {
                for_each_spaced_kmer_packed_unchecked_bytewise_scalar(
                    packed, mask_span, bit_extract_bit_loop,
                    [&](std::size_t, std::size_t, auto const&) { ++calls; }
                );
            });
            std::size_t const span_k = mask_strings.front().size();
            std::size_t const positions = seq.size() >= span_k ? seq.size() - span_k + 1 : 0;
            EXPECT_EQ(calls, positions * mask_strings.size());
        }
    }
}

// Invalid nucleotides at positions that the mask skips leave the spaced k-mer valid, in the narrow
// and in the full-width loop, and those at kept positions make it invalid.
template <Layout L>
static void check_invalid_in_gaps()
{
    WordEncoderButterfly<Encoding::kACGT, L> const encoder;
    for (std::size_t span_k : {3u, 29u, 30u, 32u}) {
        std::string const mask_string = std::string("1").append(span_k - 2, '0').append("1");
        auto const mask = make_spaced_kmer_mask<L, kNarrowMaxSpan>(mask_string);

        // Two spans with N in all of the gap, then a span of N, which reaches the kept positions of
        // all later spans.
        std::string seq = std::string("A").append(span_k - 2, 'N').append("C");
        seq += seq;
        seq.append(span_k, 'N');
        auto const packed = pack_sequence_checked(seq, encoder);

        std::vector<std::size_t> positions;
        for_each_spaced_kmer_packed_checked_bytewise_scalar(
            packed, mask, bit_extract_bit_loop,
            [&](std::size_t pos, auto const&) { positions.push_back(pos); }
        );
        EXPECT_EQ(positions, (std::vector<std::size_t>{0, span_k}));
    }
}

// =================================================================================================
//     Tests
// =================================================================================================

TEST(SpacedKmerExtractPackedBytewiseScalar, CheckedAcgtMsb)
{
    check_checked_extraction<Encoding::kACGT, Layout::kMSB>();
}

TEST(SpacedKmerExtractPackedBytewiseScalar, CheckedAcgtLsb)
{
    check_checked_extraction<Encoding::kACGT, Layout::kLSB>();
}

TEST(SpacedKmerExtractPackedBytewiseScalar, CheckedActgMsb)
{
    check_checked_extraction<Encoding::kACTG, Layout::kMSB>();
}

TEST(SpacedKmerExtractPackedBytewiseScalar, CheckedActgLsb)
{
    check_checked_extraction<Encoding::kACTG, Layout::kLSB>();
}

TEST(SpacedKmerExtractPackedBytewiseScalar, CheckedInvalidInGaps)
{
    check_invalid_in_gaps<Layout::kMSB>();
    check_invalid_in_gaps<Layout::kLSB>();
}

TEST(SpacedKmerExtractPackedBytewiseScalar, CheckedDefaultMaxMasks)
{
    check_default_max_masks<Layout::kMSB, true>();
    check_default_max_masks<Layout::kLSB, true>();
}
