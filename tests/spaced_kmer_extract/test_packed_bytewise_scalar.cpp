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
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers
// =================================================================================================

// Sequence lengths 0..100, which covers sequences shorter than the span, the last step, and the
// main loop for every span, plus two longer ones, each with random per-position content. The
// oracle (oracle.hpp) works on the ASCII input, independent of the packing and the extraction.
static std::vector<std::string> const& test_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        Splitmix64 rng(6006);
        for (std::size_t len = 0; len <= 100; ++len) {
            out.push_back(random_sequence(rng, len));
        }
        out.push_back(random_sequence(rng, 257));
        out.push_back(random_sequence(rng, 511));
        return out;
    }();
    return seqs;
}

// Random sequences of the lengths 0..160 with lower and upper case nucleotides, without invalid
// characters, with scattered ones, and with many, plus longer ones around blocks of 32 positions,
// and with runs of N of up to 100, as in assemblies, for the checked extraction.
static std::vector<std::string> const& checked_test_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        Splitmix64 rng(8008);
        for (std::size_t len = 0; len <= 160; ++len) {
            std::string seq = random_sequence(rng, len);
            double const rates[] = {0.0, 0.02, 0.2};
            inject_invalid(seq, rng, rates[len % 3], all_invalid_bytes());
            out.push_back(std::move(seq));
        }
        for (std::size_t len : {511u, 512u, 513u, 1023u, 1024u, 1025u}) {
            std::string seq = random_sequence(rng, len);
            inject_invalid(seq, rng, 0.005, kHandPickedInvalid);
            out.push_back(std::move(seq));
        }
        for (std::size_t i = 0; i < 4; ++i) {
            std::string seq = random_sequence(rng, 1000);
            for (std::size_t r = 0; r < 5; ++r) {
                std::size_t const run = 1 + rng.get_uint64() % 100;
                std::size_t const at = rng.get_uint64() % (seq.size() - run);
                seq.replace(at, run, run, 'N');
            }
            out.push_back(std::move(seq));
        }
        return out;
    }();
    return seqs;
}

// For every span in [1, 32], a single random mask, and a set of three random masks, plus masks
// that keep all positions, or only the first and the last.
static std::vector<std::vector<std::string>> const& test_mask_sets()
{
    static std::vector<std::vector<std::string>> const sets = [] {
        Splitmix64 rng(7007);
        auto random_mask = [&](std::size_t span_k) {
            std::string mask(span_k, '0');
            for (auto& c : mask) {
                c = (rng.get_uint64() & 1u) ? '1' : '0';
            }
            mask.front() = '1';
            mask.back() = '1';
            return mask;
        };

        std::vector<std::vector<std::string>> out;
        for (std::size_t span_k = 1; span_k <= 32; ++span_k) {
            out.push_back({ random_mask(span_k) });
            out.push_back({ random_mask(span_k), random_mask(span_k), random_mask(span_k) });
        }
        out.push_back({ std::string(32, '1') });
        out.push_back({ std::string(29, '1'), std::string("1").append(27, '0').append("1") });
        out.push_back({ std::string("1").append(30, '0').append("1") });
        return out;
    }();
    return sets;
}

// A mask with the given span, bypassing make_spaced_kmer_mask(), to test invalid spans.
template <Layout L>
static SpacedKmerMask<L, BitExtractMask> raw_spaced_kmer_mask(std::size_t span_k)
{
    SpacedKmerMask<L, BitExtractMask> mask;
    static_cast<BitExtractMask&>(mask) = BitExtractMask(3);
    mask.span_k = span_k;
    return mask;
}

struct NoopSpacedCallback
{
    template <typename K>
    void operator()(std::size_t, std::size_t, K const&) const {}
};

// Checks `extract(masks, callback)` against `expected`, in all callback forms, with the vector of
// masks as a span of fixed size, via dispatch_static_span(), and for a single mask also
// `extract(mask, callback)`, with the mask itself.
template <Layout L, typename Masks, typename Extract>
static void check_callback_forms(
    Masks const& masks, std::vector<SpacedEvent> const& expected, Extract&& extract
) {
    auto const runs = collect_spaced_callback_forms<L, false>([&](auto const& callback) {
        dispatch_static_span<kMaxTestMasks>(masks, [&](auto mask_span) {
            extract(mask_span, callback);
        });
    });
    for (auto const& got : runs) {
        check_spaced_events(got, expected);
    }
    if (masks.size() == 1) {
        auto const single_runs = collect_spaced_callback_forms<L, true>(
            [&](auto const& callback) {
                extract(masks.front(), callback);
            }
        );
        for (auto const& got : single_runs) {
            check_spaced_events(got, expected);
        }
    }
}

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
        auto const direct_masks = make_spaced_kmer_masks<L>(mask_strings);
        auto const block_masks = make_spaced_kmer_block_shift_masks(direct_masks);
        auto const butterfly_masks = make_spaced_kmer_butterfly_masks(direct_masks);

        for (auto const& seq : test_sequences()) {
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

// Checks the checked extraction against the oracle, which skips the spaced k-mers with an invalid
// character at a position that their mask keeps, and the unchecked one on the same sequence, which
// ignores the validity bits. The bit extraction does not depend on the validity, so that two
// extractors suffice here.
template <Encoding E, Layout L>
static void check_checked_extraction()
{
    WordEncoderButterfly<E, L> const encoder;
    for (auto const& mask_strings : test_mask_sets()) {
        auto const direct_masks = make_spaced_kmer_masks<L>(mask_strings);
        auto const butterfly_masks = make_spaced_kmer_butterfly_masks(direct_masks);

        for (auto const& seq : checked_test_sequences()) {
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
        auto const mask = make_spaced_kmer_mask<L>(mask_string);

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
//     Error Contract
// =================================================================================================

// Masks need to have the Layout of the sequence.
template <typename Seq, typename Mask>
concept CanExtractPacked = requires(Seq const& seq, Mask const& mask) {
    for_each_spaced_kmer_packed_unchecked_bytewise_scalar(
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

// Spans outside of [1, 32] and sets of masks with different spans are rejected, also for sequences
// shorter than the span.
template <Layout L>
static void check_error_contract()
{
    WordEncoderButterfly<Encoding::kACGT, L> const encoder;
    for (std::string const& seq : {std::string{}, std::string(129, 'T')}) {
        auto const packed = pack_sequence_unchecked(seq, encoder);
        for (std::size_t const span_k : {std::size_t{0}, std::size_t{33}}) {
            auto const mask = raw_spaced_kmer_mask<L>(span_k);
            EXPECT_THROW(
                for_each_spaced_kmer_packed_unchecked_bytewise_scalar(
                    packed, mask, bit_extract_bit_loop, NoopSpacedCallback{}
                ),
                std::invalid_argument
            );
            EXPECT_THROW(
                for_each_spaced_kmer_packed_unchecked_bytewise_scalar(
                    packed, std::span<SpacedKmerMask<L, BitExtractMask> const, 1>(&mask, 1),
                    bit_extract_bit_loop, NoopSpacedCallback{}
                ),
                std::invalid_argument
            );
        }

        auto const mixed = make_spaced_kmer_masks<L>({"101", "1001"});
        EXPECT_THROW(
            dispatch_static_span<kMaxTestMasks>(mixed, [&](auto mask_span) {
                for_each_spaced_kmer_packed_unchecked_bytewise_scalar(
                    packed, mask_span, bit_extract_bit_loop, NoopSpacedCallback{}
                );
            }),
            std::invalid_argument
        );
    }
}

// The largest set of masks that dispatch_static_span() hands on in the library, of
// kMaxDispatchedMasks masks, as a span of that fixed size, with one bit extractor: the other checks
// dispatch only up to kMaxTestMasks masks.
template <Layout L>
static void check_max_dispatched_masks()
{
    constexpr Encoding E = Encoding::kACGT;
    WordEncoderButterfly<E, L> const encoder;
    Splitmix64 rng(7008);
    std::vector<std::string> mask_strings;
    for (std::size_t i = 0; i < kMaxDispatchedMasks; ++i) {
        std::string mask(21, '0');
        for (auto& c : mask) {
            c = (rng.get_uint64() & 1u) ? '1' : '0';
        }
        mask.front() = '1';
        mask.back() = '1';
        mask_strings.push_back(std::move(mask));
    }
    auto const masks = make_spaced_kmer_masks<L>(mask_strings);
    std::span<SpacedKmerMask<L, BitExtractMask> const, kMaxDispatchedMasks> const mask_span(
        masks.data(), kMaxDispatchedMasks
    );

    for (auto const& seq : test_sequences()) {
        auto const packed = pack_sequence_unchecked(seq, encoder);
        std::vector<SpacedEvent> events;
        for_each_spaced_kmer_packed_unchecked_bytewise_scalar(
            packed, mask_span, bit_extract_bit_loop,
            [&](std::size_t pos, std::size_t mask_idx, SpacedKmer<E, L> const& spaced_kmer) {
                events.push_back({pos, mask_idx, kmer_value(spaced_kmer)});
            }
        );
        check_spaced_events(events, oracle_spaced_events<E, L>(seq, mask_strings));
    }
}

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

TEST(SpacedKmerExtractPackedBytewiseScalar, MaxDispatchedMasks)
{
    check_max_dispatched_masks<Layout::kMSB>();
    check_max_dispatched_masks<Layout::kLSB>();
}

TEST(SpacedKmerExtractPackedBytewiseScalar, ErrorContract)
{
    check_error_contract<Layout::kMSB>();
    check_error_contract<Layout::kLSB>();
}
