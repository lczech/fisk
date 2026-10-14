#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/core/random.hpp"
#include "fisk/kmer_spaced/kmer_spaced.hpp"
#include "fisk/kmer_spaced/packed.hpp"
#include "fisk/seq_pack/seq_pack.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers
// =================================================================================================

// Sequence lengths 0..100, which covers sequences shorter than the span, the tail, and the main
// loop for every span, plus two longer ones, each with random per-position content. The oracle
// (oracle.hpp) works on the ASCII input, independent of the packing and the extraction.
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
        out.push_back({ std::string(29, '1'), "1" + std::string(27, '0') + "1" });
        out.push_back({ "1" + std::string(30, '0') + "1" });
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

// =================================================================================================
//     Extraction
// =================================================================================================

// Checks every bit extractor, through the overload for a vector of masks, and, for a single mask,
// also through the single-mask overload, in all callback forms, against the oracle.
template <Encoding E, Layout L>
static void check_extraction()
{
    WordEncoderButterfly<E, L> const encoder;
    for (auto const& mask_strings : test_mask_sets()) {
        auto const direct_masks = make_spaced_kmer_masks<L>(mask_strings);
        auto const block_masks = make_spaced_kmer_block_tables(direct_masks);
        auto const butterfly_masks = make_spaced_kmer_butterfly_tables(direct_masks);

        for (auto const& seq : test_sequences()) {
            auto const packed = pack_sequence(seq, encoder);
            auto const expected = oracle_spaced_events<E, L>(seq, mask_strings);

            auto const check = [&](auto const& masks, auto const& bit_ext) {
                auto const runs = collect_spaced_callback_forms<L, false>(
                    [&](auto const& callback) {
                        for_each_spaced_kmer_packed(packed, masks, bit_ext, callback);
                    }
                );
                for (auto const& got : runs) {
                    check_spaced_events(got, expected);
                }
                if (masks.size() == 1) {
                    auto const single_runs = collect_spaced_callback_forms<L, true>(
                        [&](auto const& callback) {
                            for_each_spaced_kmer_packed(packed, masks.front(), bit_ext, callback);
                        }
                    );
                    for (auto const& got : single_runs) {
                        check_spaced_events(got, expected);
                    }
                }
            };

            check(direct_masks, [](std::uint64_t x, BitExtractMask const& mask) {
                return bit_extract_bitloop(x, mask);
            });
            check(direct_masks, [](std::uint64_t x, BitExtractMask const& mask) {
                return bit_extract_byte_table(x, mask);
            });
            check(block_masks, [](std::uint64_t x, BitExtractBlockTable const& mask) {
                return bit_extract_block_table(x, mask);
            });
            check(block_masks, [](std::uint64_t x, BitExtractBlockTable const& mask) {
                return bit_extract_block_table_unrolled<8>(x, mask);
            });
            check(butterfly_masks, [](std::uint64_t x, BitExtractButterflyTable const& mask) {
                return bit_extract_butterfly_table(x, mask);
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

// Masks need to have the Layout of the sequence.
template <typename Seq, typename Mask>
concept CanExtractPacked = requires(Seq const& seq, Mask const& mask) {
    for_each_spaced_kmer_packed(seq, mask, bit_extract_bitloop, NoopSpacedCallback{});
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
// shorter than the span, and an empty set of masks emits nothing.
template <Layout L>
static void check_error_contract()
{
    WordEncoderButterfly<Encoding::kACGT, L> const encoder;
    for (std::string const& seq : {std::string{}, std::string(129, 'T')}) {
        auto const packed = pack_sequence(seq, encoder);
        for (std::size_t const span_k : {std::size_t{0}, std::size_t{33}}) {
            auto const mask = raw_spaced_kmer_mask<L>(span_k);
            EXPECT_THROW(
                for_each_spaced_kmer_packed(
                    packed, mask, bit_extract_bitloop, NoopSpacedCallback{}
                ),
                std::invalid_argument
            );
            EXPECT_THROW(
                for_each_spaced_kmer_packed(
                    packed, std::vector{mask}, bit_extract_bitloop, NoopSpacedCallback{}
                ),
                std::invalid_argument
            );
        }

        auto const mixed = make_spaced_kmer_masks<L>({"101", "1001"});
        EXPECT_THROW(
            for_each_spaced_kmer_packed(packed, mixed, bit_extract_bitloop, NoopSpacedCallback{}),
            std::invalid_argument
        );

        std::size_t calls = 0;
        for_each_spaced_kmer_packed(
            packed, std::vector<SpacedKmerMask<L, BitExtractMask>>{}, bit_extract_bitloop,
            [&](std::size_t, std::size_t, auto const&) { ++calls; }
        );
        EXPECT_EQ(calls, std::size_t{0});
    }
}

// =================================================================================================
//     Tests
// =================================================================================================

TEST(KmerSpacedPacked, AcgtMsb)
{
    check_extraction<Encoding::kACGT, Layout::kMSB>();
}

TEST(KmerSpacedPacked, AcgtLsb)
{
    check_extraction<Encoding::kACGT, Layout::kLSB>();
}

TEST(KmerSpacedPacked, ActgMsb)
{
    check_extraction<Encoding::kACTG, Layout::kMSB>();
}

TEST(KmerSpacedPacked, ActgLsb)
{
    check_extraction<Encoding::kACTG, Layout::kLSB>();
}

TEST(KmerSpacedPacked, ErrorContract)
{
    check_error_contract<Layout::kMSB>();
    check_error_contract<Layout::kLSB>();
}
