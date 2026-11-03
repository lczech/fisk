#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/util/random.hpp"
#include "fisk/core/types.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"
#include "fisk/spaced_kmer_extract/packed_bytewise_scalar.hpp"
#include "corpus.hpp"
#include "spaced_kmer_extract/common.hpp"
#include "spaced_kmer_extract/packed_scalar_common.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers
// =================================================================================================

struct NoopSpacedCallback
{
    template <typename K>
    void operator()(std::size_t, std::size_t, K const&) const {}
};

// =================================================================================================
//     Extraction
// =================================================================================================

// Checks every bit extractor, through the overload for a set of masks, and, for a single mask, also
// through the single-mask overload, in all callback forms, against the oracle.
template <Encoding E, Layout L>
static void check_extraction()
{
    WordEncoderButterfly<E, L> const encoder;
    for (auto const& mask_strings : test_mask_sets()) {
        auto const direct_masks = make_spaced_kmer_masks<L, kNarrowMaxSpan>(mask_strings);
        auto const block_masks = make_spaced_kmer_block_shift_masks(direct_masks);
        auto const butterfly_masks = make_spaced_kmer_butterfly_masks(direct_masks);

        for (auto const& seq : packed_test_sequences()) {
            auto const packed = pack_sequence_unchecked(seq, encoder);
            auto const expected = oracle_spaced_events<E, L>(seq, mask_strings);

            auto const check = [&](auto const& masks, auto const& bit_ext) {
                check_callback_forms<L>(
                    masks, expected, [&](auto const& mask_or_masks, auto const& callback) {
                        for_each_spaced_kmer_packed_unchecked_bytewise_scalar(
                            packed, mask_or_masks, bit_ext, callback
                        );
                    }
                );
            };

            check(direct_masks, [](std::uint64_t x, BitExtractMask const& mask) {
                return bit_extract_bit_loop(x, mask);
            });
            check(direct_masks, [](std::uint64_t x, BitExtractMask const& mask) {
                return bit_extract_byte_lookup(x, mask);
            });
            check(block_masks, [](std::uint64_t x, BitExtractBlockShiftMask const& mask) {
                return bit_extract_block_shift(x, mask);
            });
            check(block_masks, [](std::uint64_t x, BitExtractBlockShiftMask const& mask) {
                return bit_extract_block_shift_unrolled<8>(x, mask);
            });
            check(butterfly_masks, [](std::uint64_t x, BitExtractButterflyMask const& mask) {
                return bit_extract_butterfly(x, mask);
            });
            #if defined(FISK_HAS_BMI2)
            check(direct_masks, [](std::uint64_t x, BitExtractMask const& mask) {
                return bit_extract_pext(x, mask);
            });
            #endif
        }
    }
}

// =================================================================================================
//     Error Contract
// =================================================================================================

// Masks need to have the Layout of the sequence, unchecked and checked.
template <typename Seq, typename Mask>
concept CanExtractPacked = requires(Seq const& seq, Mask const& mask) {
    for_each_spaced_kmer_packed_unchecked_bytewise_scalar(
        seq, mask, bit_extract_bit_loop, NoopSpacedCallback{}
    );
};
template <typename Seq, typename Mask>
concept CanExtractPackedChecked = requires(Seq const& seq, Mask const& mask) {
    for_each_spaced_kmer_packed_checked_bytewise_scalar(
        seq, mask, bit_extract_bit_loop, NoopSpacedCallback{}
    );
};

static_assert(CanExtractPacked<
    PackedSequence<Encoding::kACGT, Layout::kMSB>, SpacedKmerMask<Layout::kMSB, BitExtractMask>
>);
static_assert(CanExtractPacked<
    PackedSequence<Encoding::kACTG, Layout::kLSB>, SpacedKmerMask<Layout::kLSB, BitExtractMask>
>);
static_assert(!CanExtractPacked<
    PackedSequence<Encoding::kACGT, Layout::kMSB>, SpacedKmerMask<Layout::kLSB, BitExtractMask>
>);
static_assert(!CanExtractPacked<
    PackedSequence<Encoding::kACGT, Layout::kLSB>, SpacedKmerMask<Layout::kMSB, BitExtractMask>
>);
static_assert(!CanExtractPacked<PackedSequence<Encoding::kACGT, Layout::kMSB>, BitExtractMask>);

static_assert(CanExtractPackedChecked<
    PackedSequenceWithValidity<Encoding::kACGT, Layout::kMSB>,
    SpacedKmerMask<Layout::kMSB, BitExtractMask>
>);
static_assert(CanExtractPackedChecked<
    PackedSequenceWithValidity<Encoding::kACTG, Layout::kLSB>,
    SpacedKmerMask<Layout::kLSB, BitExtractMask>
>);
static_assert(!CanExtractPackedChecked<
    PackedSequenceWithValidity<Encoding::kACGT, Layout::kMSB>,
    SpacedKmerMask<Layout::kLSB, BitExtractMask>
>);
static_assert(!CanExtractPackedChecked<
    PackedSequenceWithValidity<Encoding::kACGT, Layout::kLSB>,
    SpacedKmerMask<Layout::kMSB, BitExtractMask>
>);
static_assert(!CanExtractPackedChecked<
    PackedSequenceWithValidity<Encoding::kACGT, Layout::kMSB>, BitExtractMask
>);

// =================================================================================================
//     Tests
// =================================================================================================

TEST(SpacedKmerExtractPackedBytewiseScalar, UncheckedAcgtMsb)
{
    check_extraction<Encoding::kACGT, Layout::kMSB>();
}

TEST(SpacedKmerExtractPackedBytewiseScalar, UncheckedAcgtLsb)
{
    check_extraction<Encoding::kACGT, Layout::kLSB>();
}

TEST(SpacedKmerExtractPackedBytewiseScalar, UncheckedActgMsb)
{
    check_extraction<Encoding::kACTG, Layout::kMSB>();
}

TEST(SpacedKmerExtractPackedBytewiseScalar, UncheckedActgLsb)
{
    check_extraction<Encoding::kACTG, Layout::kLSB>();
}

TEST(SpacedKmerExtractPackedBytewiseScalar, UncheckedDefaultMaxMasks)
{
    check_default_max_masks<Layout::kMSB, false>();
    check_default_max_masks<Layout::kLSB, false>();
}

TEST(SpacedKmerExtractPackedBytewiseScalar, ErrorContract)
{
    check_error_contract<Layout::kMSB, kNarrowMaxSpan>();
    check_error_contract<Layout::kLSB, kNarrowMaxSpan>();
}
