#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "fisk/core/cast.hpp"
#include "fisk/util/random.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/packed_bytewise_scalar.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"
#include "corpus.hpp"
#include "oracle.hpp"

// =================================================================================================
//     Extraction from ASCII
// =================================================================================================

// The checks of the extraction from ASCII take the loop under test as a callable `extract`, called
// as `extract.template operator()<E, L>(seq, k, callback)` for the k-mers of the conventions E and
// L. A loop that takes an encoder runs with the lookup table encoder of E. The loops that take an
// encoder are also checked with other encoders, by a callable `extract_with`, called as
// `extract_with.template operator()<L>(seq, k, encoder, callback)`.

// -------------------------------------------------------------------------------------------------
//     Helpers
// -------------------------------------------------------------------------------------------------

// Checks `extract` against the oracle for one (seq, k) under the conventions E and L: exact
// positions and values, in order, skipping every window that overlaps an invalid character, in
// both callback forms.
template <fisk::Encoding E, fisk::Layout L, typename Extract>
inline void check_ascii_extractor_in(std::string const& seq, std::size_t k, Extract const& extract)
{
    auto const expected = oracle_kmers_skipping_invalid<E, L>(seq, k);
    check_kmer_callbacks<E, L>(expected, k, [&](auto const& callback) {
        extract.template operator()<E, L>(seq, k, callback);
    });
}

// Same, under all four conventions.
template <typename Extract>
inline void check_ascii_extractor(std::string const& seq, std::size_t k, Extract const& extract)
{
    using fisk::Encoding;
    using fisk::Layout;
    check_ascii_extractor_in<Encoding::kACGT, Layout::kMSB>(seq, k, extract);
    check_ascii_extractor_in<Encoding::kACGT, Layout::kLSB>(seq, k, extract);
    check_ascii_extractor_in<Encoding::kACTG, Layout::kMSB>(seq, k, extract);
    check_ascii_extractor_in<Encoding::kACTG, Layout::kLSB>(seq, k, extract);
}

// Checks `extract_with` with the encoder `enc` against the oracle for one (seq, k), in both
// layouts and both callback forms. `as_seen` is `seq` as the encoder sees it, for the oracle; it
// differs from `seq` only for an encoder that deems other characters invalid than the oracle does.
template <typename Enc, typename ExtractWith>
inline void check_ascii_extractor_with(
    std::string const& seq, std::size_t k, Enc const& enc, std::string const& as_seen,
    ExtractWith const& extract_with
) {
    constexpr auto E = Enc::encoding;
    auto const check_in = [&]<fisk::Layout L>() {
        auto const expected = oracle_kmers_skipping_invalid<E, L>(as_seen, k);
        check_kmer_callbacks<E, L>(expected, k, [&](auto const& callback) {
            extract_with.template operator()<L>(seq, k, enc, callback);
        });
    };
    check_in.template operator()<fisk::Layout::kMSB>();
    check_in.template operator()<fisk::Layout::kLSB>();
}

// Valid-only sequences (pure ACGT/acgt), shared across the oracle sweep and the invalid-sentinel
// check below (whose custom encoder repurposes the lowercase half of this alphabet as "invalid").
// Lengths 0..70 plus a batch of longer random lengths.
inline std::vector<std::string> const& valid_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        fisk::Splitmix64 rng(90210);
        for (std::size_t len = 0; len <= 70; ++len) {
            out.push_back(random_sequence(rng, len));
        }
        for (int i = 0; i < 20; ++i) {
            std::size_t const len = static_cast<std::size_t>(rng.get_uint64() % 300);
            out.push_back(random_sequence(rng, len));
        }
        return out;
    }();
    return seqs;
}

// -------------------------------------------------------------------------------------------------
//     Oracle Correctness
// -------------------------------------------------------------------------------------------------

// Correct k-mers, correct positions, correct order, correct encoding and layout, for every
// length/k combination in the shared sequence set.
template <typename Extract>
inline void check_ascii_oracle(Extract const& extract)
{
    for (auto const& seq : valid_sequences()) {
        for (auto const k : test_ks()) {
            check_ascii_extractor(seq, k, extract);
        }
    }
}

// -------------------------------------------------------------------------------------------------
//     k and Length Boundaries
// -------------------------------------------------------------------------------------------------

// Empty input, and sequences one shorter than / exactly / one longer than k: zero, one, and two
// k-mers respectively, with no out-of-bounds access at any of these edges.
template <typename Extract>
inline void check_ascii_length_boundaries(Extract const& extract)
{
    std::string seq;
    while (seq.size() < 40) {
        seq += "ACGT";
    }

    for (auto const k : test_ks()) {
        check_ascii_extractor("", k, extract);
        check_ascii_extractor(seq.substr(0, k - 1), k, extract);
        check_ascii_extractor(seq.substr(0, k), k, extract);
        check_ascii_extractor(seq.substr(0, k + 1), k, extract);
    }
}

// -------------------------------------------------------------------------------------------------
//     Invalid Characters
// -------------------------------------------------------------------------------------------------

// Hand-crafted placements: invalid at the very start/middle/end, a run of several in a row,
// alternating valid/invalid, and a sequence that is entirely invalid. Every k-mer overlapping an
// invalid symbol must be suppressed, and the window must fully refill before emitting again.
// Written with 'N' for readability, and then checked with every hand-picked invalid character in
// its place.
template <typename Extract>
inline void check_ascii_invalid_character_placement(Extract const& extract)
{
    std::vector<std::string> const seqs = {
        "NACGTACGTACGT",
        "ACGTACGNTACGT",
        "ACGTACGTACGTN",
        "ACGTNNNACGTACGT",
        "ANCNGNTNACNGNTN",
        "NNNNNNNNNNNNNNN",
        "N",
        "A",
        // Longer placements, so that k=16 and k=32 below exercise invalid-run-skipping and the
        // window refill too, rather than degenerating into a "sequence shorter than k" check.
        std::string("N").append(45, 'A'),
        std::string(45, 'A') + "N",
        std::string(20, 'A') + "N" + std::string(20, 'A'),
        std::string(20, 'A') + "NNN" + std::string(20, 'A'),
    };
    std::vector<std::size_t> const ks = { 1, 3, 4, 8, 16, 32 };

    for (auto const marker : kHandPickedInvalid) {
        for (auto seq : seqs) {
            std::replace(seq.begin(), seq.end(), 'N', marker);
            for (auto const k : ks) {
                check_ascii_extractor(seq, k, extract);
            }
        }
    }
}

// Custom encoder whose invalid sentinel is not exactly 4 (255 here, and lowercase bases are
// deliberately "invalid" under it), for the loops that take an encoder, which honor the documented
// ">= 4 is invalid" contract rather than special-casing the value 4. Also shows the shape a
// user-defined encoder takes: a functor that states its encoding.
struct SentinelNotFourEncoder
{
    static constexpr fisk::Encoding encoding = fisk::Encoding::kACGT;

    constexpr std::uint8_t operator()(char c) const noexcept
    {
        switch (c) {
            case 'A': return 0;
            case 'C': return 1;
            case 'G': return 2;
            case 'T': return 3;
            default:  return 255;
        }
    }
};

// Checks `extract_with` with SentinelNotFourEncoder on the valid-only sequences, whose lowercase
// bases it deems invalid.
template <typename ExtractWith>
inline void check_ascii_invalid_sentinel_not_four(ExtractWith const& extract_with)
{
    SentinelNotFourEncoder const enc;
    for (auto const& seq : valid_sequences()) {
        // What the encoder sees as invalid, spelled out for the oracle: every lowercase base.
        std::string as_seen = seq;
        for (auto& c : as_seen) {
            if (std::islower(static_cast<unsigned char>(c))) {
                c = 'N';
            }
        }
        for (auto const k : test_ks()) {
            check_ascii_extractor_with(seq, k, enc, as_seen, extract_with);
        }
    }
}

// Randomized sequences with invalid characters from the full byte range injected at a controlled
// rate, swept across sparse, moderate, and heavy regimes.
template <typename Extract>
inline void check_ascii_invalid_injection_fuzz(Extract const& extract)
{
    struct Rate { double p; std::uint64_t seed; };
    std::vector<Rate> const rates = {
        {0.02, 111111},
        {0.10, 222222},
        {0.30, 333333},
    };

    for (auto const& r : rates) {
        for (auto const& seq : invalid_injected_sequences(40, 149, r.p, r.seed)) {
            for (auto const k : test_ks()) {
                check_ascii_extractor(seq, k, extract);
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
//     Block Boundaries
// -------------------------------------------------------------------------------------------------

// for_each_kmer_ascii_checked_rolling_avx2() processes 32-character AVX2 blocks (falling back to a
// scalar tail), and for_each_kmer_ascii_checked_rolling_swar() processes 8-character blocks; both
// then finish with a byte-at-a-time remainder. 16 and 64 are included pre-emptively, matching the
// SSE2/AVX512 block widths already used elsewhere in the codebase, so this generator needs no
// changes if k-mer extraction grows those variants too. The loops without blocks are checked on
// the same sequences.
//
// For each block size, builds sequences one-under/on/one-over one and two block widths, with an
// invalid marker at the last character of a block, the first character of the next, or straddling
// both -- exactly where a block-boundary bug would hide. The marker cycles through the hand-picked
// invalid characters from one sequence to the next.
template <typename Extract>
inline void check_ascii_block_boundaries(Extract const& extract)
{
    std::vector<std::size_t> const block_sizes = { 8, 16, 32, 64 };
    fisk::Splitmix64 rng(424242);
    std::size_t marker_idx = 0;

    for (auto const bs : block_sizes) {
        std::vector<std::size_t> const lens = {
            1 * bs - 1, 1 * bs, 1 * bs + 1,
            2 * bs - 1, 2 * bs, 2 * bs + 1,
            3 * bs - 1, 3 * bs, 3 * bs + 1,
            4 * bs - 1, 4 * bs, 4 * bs + 1
        };
        std::vector<std::vector<std::size_t>> const marker_sets = {
            { bs - 1 },             // last char of the first block
            { bs },                 // first char of the second block
            { bs - 1, bs },         // straddling the boundary
            { bs - 2, bs - 1, bs, bs + 1 }, // a run straddling the boundary
        };

        for (auto const len : lens) {
            std::string const base = random_sequence(rng, len, "ACGT");

            for (auto const& markers : marker_sets) {
                std::string seq = base;
                char const marker = kHandPickedInvalid[marker_idx++ % kHandPickedInvalid.size()];
                bool any_in_range = false;
                for (auto const pos : markers) {
                    if (pos < seq.size()) {
                        seq[pos] = marker;
                        any_in_range = true;
                    }
                }
                if (!any_in_range) {
                    continue;
                }

                for (auto const k : test_ks()) {
                    check_ascii_extractor(seq, k, extract);
                }
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
//     Case Insensitivity
// -------------------------------------------------------------------------------------------------

// A single deterministic, easy-to-eyeball mixed-case sequence, checked end-to-end. Per-character
// case handling is already exhaustively covered in core/test_char_encoder.cpp; this just confirms
// it survives all the way through k-mer assembly.
template <typename Extract>
inline void check_ascii_mixed_case(Extract const& extract)
{
    std::string const seq = "AcGtACgtacGTAcGtACgtacGTAcGtACgt";
    std::vector<std::size_t> const ks = { 1, 4, 8, 16, 32 };
    for (auto const k : ks) {
        check_ascii_extractor(seq, k, extract);
    }
}

// =================================================================================================
//     Extraction from a PackedSequence
// =================================================================================================

// The oracle (oracle.hpp) decodes directly from the ASCII input, independent of the packed
// extractors under test; the packing step itself is checked in sequence_pack/.

// -------------------------------------------------------------------------------------------------
//     Unchecked
// -------------------------------------------------------------------------------------------------

// Sequence lengths 0..512, each with random per-position content.
inline std::vector<std::string> const& packed_test_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        fisk::Splitmix64 rng(5005);
        for (std::size_t len = 0; len <= 512; ++len) {
            out.push_back(random_sequence(rng, len));
        }
        return out;
    }();
    return seqs;
}

// Checks one unchecked packed extractor, called as `extract(packed, k, callback)`, under the
// conventions of E and L: the complete ordered output, positions and values, in both callback
// forms, for every k and every test sequence, plus its invalid-k contract.
template <fisk::Encoding E, fisk::Layout L, typename Extract>
inline void check_packed_variant(Extract const& extract)
{
    fisk::WordEncoderButterfly<E, L> const encoder;
    sweep_packed_extractors(
        encoder, packed_test_sequences(), 32,
        [&](std::string const&, auto const& packed, std::size_t k, auto const& expected) {
            check_kmer_callbacks<E, L>(expected, k, [&](auto const& callback) {
                extract(packed, k, callback);
            });
        }
    );
    check_invalid_k_throws(32, [&](std::string const& seq, std::size_t k) {
        extract(fisk::pack_sequence_unchecked(seq, encoder), k, [](fisk::Kmer<E, L>) {});
    });
}

// -------------------------------------------------------------------------------------------------
//     Checked
// -------------------------------------------------------------------------------------------------

// The checks of the checked extraction take the loop under test as a callable `check`, called as
// `check(packed, k, expected)`, which runs the loop on the PackedSequenceWithValidity `packed` for
// k-mers of length `k`, and checks its k-mers against `expected`.

// Random sequences of the lengths 0..300 with lower and upper case nucleotides, without invalid
// characters, with scattered ones, and with many. Plus longer ones around blocks of 32 and 64
// positions, and with runs of N of up to 100, as in assemblies.
inline std::vector<std::string> const& checked_test_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        fisk::Splitmix64 rng(7007);
        for (std::size_t len = 0; len <= 300; ++len) {
            std::string seq = random_sequence(rng, len);
            double const rates[] = {0.0, 0.02, 0.2};
            inject_invalid(seq, rng, rates[len % 3], all_invalid_bytes());
            out.push_back(std::move(seq));
        }
        std::size_t const long_lengths[] = {511, 512, 513, 1000, 1023, 1024, 1025};
        for (std::size_t len : long_lengths) {
            std::string seq = random_sequence(rng, len);
            inject_invalid(seq, rng, 0.005, kHandPickedInvalid);
            out.push_back(std::move(seq));
        }
        for (std::size_t i = 0; i < 8; ++i) {
            std::string seq = random_sequence(rng, 2000);
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

// Runs `check` on `packed`, for all k, against `expected_of(k)`.
template <fisk::Encoding E, fisk::Layout L, typename ExpectedOf, typename Check>
inline void check_checked_extractor(
    fisk::PackedSequenceWithValidity<E, L> const& packed, ExpectedOf&& expected_of,
    Check const& check
) {
    for (std::size_t k = 1; k <= 32; ++k) {
        check(packed, k, expected_of(k));
    }
}

// A `check` for a checked extractor that calls back with single k-mers, called as
// `extract(packed, k, callback)`, in both callback forms.
template <typename Extract>
inline auto checked_kmers_check(Extract const& extract)
{
    return [extract](auto const& packed, std::size_t k, std::vector<ExpectedKmer> const& expected) {
        using Seq = std::remove_cvref_t<decltype(packed)>;
        check_kmer_callbacks<Seq::encoding, Seq::layout>(expected, k, [&](auto const& callback) {
            extract(packed, k, callback);
        });
    };
}

// Runs `check` on `seq`, packed with its validity, against the k-mers of its windows without
// invalid characters.
template <fisk::Encoding E, fisk::Layout L, typename Check>
inline void check_checked_sequence(std::string const& seq, Check const& check)
{
    auto const packed = fisk::pack_sequence_checked(seq, fisk::WordEncoderButterfly<E, L>{});
    check_checked_extractor(packed, [&](std::size_t k) {
        return oracle_kmers_skipping_invalid<E, L>(seq, k);
    }, check);
}

template <fisk::Encoding E, fisk::Layout L, typename Check>
inline void check_checked_random_sequences(Check const& check)
{
    for (auto const& seq : checked_test_sequences()) {
        check_checked_sequence<E, L>(seq, check);
    }
}

// One N at each position, on lengths that span two blocks of 32 plus the k - 1 nucleotides after
// them: its first and last nucleotides, and the edges of blocks and of the windows of their test.
template <fisk::Encoding E, fisk::Layout L, typename Check>
inline void check_checked_single_invalid(Check const& check)
{
    fisk::Splitmix64 rng(8008);
    std::size_t const lengths[] = {64, 97, 130};
    for (std::size_t len : lengths) {
        std::string const base = random_sequence(rng, len);
        for (std::size_t at = 0; at < len; ++at) {
            std::string seq = base;
            seq[at] = 'N';
            check_checked_sequence<E, L>(seq, check);
        }
    }
}

// A run of valid nucleotides of each length up to 40 between two Ns, so that the k-mers in the
// run are valid just for k up to its length, at offsets around the edges of the first block.
template <fisk::Encoding E, fisk::Layout L, typename Check>
inline void check_checked_valid_runs(Check const& check)
{
    fisk::Splitmix64 rng(9009);
    std::size_t const offsets[] = {0, 1, 20, 30, 31, 32, 33};
    for (std::size_t run = 0; run <= 40; ++run) {
        for (std::size_t offset : offsets) {
            std::string const seq = random_sequence(rng, offset) + "N" + random_sequence(rng, run)
                + "N" + random_sequence(rng, 40);
            check_checked_sequence<E, L>(seq, check);
        }
    }
}

// With all validity bits set, the checked extractors emit the same k-mers as the unchecked ones,
// whatever the characters were.
template <fisk::Encoding E, fisk::Layout L, typename Check>
inline void check_checked_all_valid_equals_unchecked(Check const& check)
{
    for (auto const& seq : checked_test_sequences()) {
        auto packed = fisk::pack_sequence_checked(seq, fisk::WordEncoderButterfly<E, L>{});
        auto const validity = packed.validity();
        for (std::size_t i = 0; i < packed.length(); ++i) {
            validity[i / 8] = static_cast<std::uint8_t>(validity[i / 8] | (1u << (i % 8)));
        }
        check_checked_extractor(packed, [&](std::size_t k) {
            std::vector<ExpectedKmer> expected;
            fisk::for_each_kmer_packed_unchecked_bytewise_scalar(
                packed, k, [&](std::size_t pos, fisk::Kmer<E, L> kmer) {
                    expected.push_back({pos, fisk::kmer_value(kmer)});
                }
            );
            return expected;
        }, check);
    }
}

// A nucleotide that is invalid only by its validity bit is invalid for the checked extractors: they
// never look at the codes for validity. The unchecked extractor of the same algorithm,
// `unchecked`, called as `unchecked(packed, k, callback)` with single k-mers, takes the same
// sequence, and ignores its validity bits.
template <fisk::Encoding E, fisk::Layout L, typename Check, typename Unchecked>
inline void check_checked_invalid_in_validity_only(Check const& check, Unchecked const& unchecked)
{
    fisk::Splitmix64 rng(9009);
    std::string const seq = random_sequence(rng, 200);
    std::size_t const invalid_at[] = {0, 1, 31, 32, 63, 64, 95, 96, 150, 199};
    for (std::size_t at : invalid_at) {
        auto packed = fisk::pack_sequence_checked(seq, fisk::WordEncoderButterfly<E, L>{});
        auto const validity = packed.validity();
        validity[at / 8] = static_cast<std::uint8_t>(validity[at / 8] & ~(1u << (at % 8)));
        std::string marked = seq;
        marked[at] = 'N';
        check_checked_extractor(packed, [&](std::size_t k) {
            return oracle_kmers_skipping_invalid<E, L>(marked, k);
        }, check);

        for (std::size_t k = 1; k <= 32; ++k) {
            auto const expected = oracle_kmers_skipping_invalid<E, L>(seq, k);
            check_kmer_callbacks<E, L>(expected, k, [&](auto const& callback) {
                unchecked(packed, k, callback);
            });
        }
    }
}

// An invalid k throws, for a checked extractor called as `extract(packed, k)`.
template <fisk::Encoding E, fisk::Layout L, typename Extract>
inline void check_checked_throws(Extract const& extract)
{
    check_invalid_k_throws(32, [&](std::string const& seq, std::size_t k) {
        extract(fisk::pack_sequence_checked(seq, fisk::WordEncoderButterfly<E, L>{}), k);
    });
}

// =================================================================================================
//     Chunked Extraction from ASCII
// =================================================================================================

// The chunked loops pack the input chunk by chunk, see kmer_extract/ascii_chunked.hpp. They are
// checked through a callable `extract`, called as
// `extract.template operator()<ChunkSize>(seq, k, encoder, scratch, callback)`, where `scratch`
// points to the scratch to use, or is null for the overload with the thread-local buffer, and
// `callback` takes single k-mers, in both forms; the loops that call back with vectors hand on
// their lanes with for_each_valid_lane(). The oracle decodes directly from the ASCII input, since
// these checks are also meant to exercise the packing step.

// Sequence lengths 0..300, each with random per-position content -- deliberately covering many
// exact multiples, one-more, and one-less relationships against every chunk size in
// for_each_chunk_size() below, rather than hand-picking a handful of specific (length, chunk
// size) pairs.
inline std::vector<std::string> const& chunked_test_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        fisk::Splitmix64 rng(90125);
        for (std::size_t len = 0; len <= 300; ++len) {
            out.push_back(random_sequence(rng, len));
        }
        return out;
    }();
    return seqs;
}

// Smaller sweeps for ACTG: these exist to isolate the Encoding axis on its own, not to re-cover
// every (length, k, chunk size) combination the ACGT ones already do.
inline std::vector<std::string> const& chunked_actg_test_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        for (std::size_t const len : std::vector<std::size_t>{0, 1, 31, 63, 64, 200}) {
            out.push_back(chunked_test_sequences()[len]);
        }
        return out;
    }();
    return seqs;
}

// Sequences with invalid characters: hand-picked characters at the start, end, and around the
// word and chunk boundaries of the packing step, plus random bytes from the full range. The
// unchecked loops silently encode each as some nucleotide, which must only change the k-mers
// overlapping it, not their count or positions, nor any other k-mer; the checked loops skip the
// k-mers overlapping it, also across the ends of the chunks.
inline std::vector<std::string> const& chunked_invalid_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        fisk::Splitmix64 rng(90126);
        std::size_t marker_idx = 0;
        for (std::size_t const len : std::vector<std::size_t>{1, 8, 33, 100}) {
            for (std::size_t const pos : {std::size_t{0}, len / 2, len - 1}) {
                std::string seq = random_sequence(rng, len);
                seq[pos] = kHandPickedInvalid[marker_idx++ % kHandPickedInvalid.size()];
                out.push_back(std::move(seq));
            }
        }
        for (auto& seq : invalid_injected_sequences(30, 150, 0.05, 90127)) {
            out.push_back(std::move(seq));
        }
        return out;
    }();
    return seqs;
}

// Calls `f.template operator()<ChunkSize>()` for chunk sizes spanning: smaller than most k
// (each chunk then emits fewer k-mers than it looks ahead by), around the blocks of 32 positions
// of the packed loops, and much larger than the longest test sequence (degenerating to a single
// chunk). Few of them, as the chunked loops are inlined into each chunk size, which makes them
// expensive to compile.
template <typename F>
inline void for_each_chunk_size(F&& f)
{
    [&]<std::size_t... Sizes>(std::index_sequence<Sizes...>) {
        (f.template operator()<Sizes>(), ...);
    }(std::index_sequence<1, 7, 31, 32, 33, 1000>{});
}

// The callbacks of the checks below, in both forms, as named types rather than lambdas: the
// lambdas of a check would have a type of their own for each chunk size, which would compile the
// packed loops once per chunk size.
template <fisk::Encoding E, fisk::Layout L>
struct ChunkedKmerCollector
{
    std::vector<std::size_t>* positions;
    std::vector<std::uint64_t>* values;

    void operator()(std::size_t pos, fisk::Kmer<E, L> kmer) const
    {
        positions->push_back(pos);
        values->push_back(fisk::kmer_value(kmer));
    }
};

template <fisk::Encoding E, fisk::Layout L>
struct ChunkedKmerValueCollector
{
    std::vector<std::uint64_t>* values;

    void operator()(fisk::Kmer<E, L> kmer) const
    {
        values->push_back(fisk::kmer_value(kmer));
    }
};

// Checks `extract` with the word encoder of the given conventions, for every sequence in `seqs`,
// every valid k, and every chunk size, with a scratch and with the thread-local buffer, in both
// callback forms. Unchecked, against every window: exact positions, and exact values for every
// window that consists of nucleotides only. Checked, against the windows that consist of
// nucleotides only.
template <fisk::Encoding E, fisk::Layout L, bool Checked, typename Extract>
inline void check_ascii_chunked(std::vector<std::string> const& seqs, Extract const& extract)
{
    using Scratch = std::conditional_t<
        Checked, fisk::PackedSequenceWithValidity<E, L>, fisk::PackedSequence<E, L>
    >;
    fisk::WordEncoderButterfly<E, L> const encoder;
    Scratch scratch;
    std::vector<std::size_t> positions;
    std::vector<std::uint64_t> values;
    for (auto const& seq : seqs) {
        for (std::size_t k = 1; k <= 32; ++k) {
            auto const expected = Checked
                ? oracle_kmers_skipping_invalid<E, L>(seq, k)
                : oracle_kmers_all_windows<E, L>(seq, k);
            for_each_chunk_size([&]<std::size_t ChunkSize>() {
                for (Scratch* const s : {static_cast<Scratch*>(nullptr), &scratch}) {
                    positions.clear();
                    values.clear();
                    extract.template operator()<ChunkSize>(
                        seq, k, encoder, s, ChunkedKmerCollector<E, L>{&positions, &values}
                    );
                    check_emitted_kmers(expected, k, values, &positions);

                    values.clear();
                    extract.template operator()<ChunkSize>(
                        seq, k, encoder, s, ChunkedKmerValueCollector<E, L>{&values}
                    );
                    check_emitted_kmers(expected, k, values, nullptr);
                }
            });
        }
    }
}

// The checks of a chunked loop: the sweep above under all four conventions, on random and on
// invalid sequences, and an invalid k throws, with and without a scratch.
template <bool Checked, typename Extract>
inline void check_ascii_chunked_all(Extract const& extract)
{
    using fisk::Encoding;
    using fisk::Layout;
    check_ascii_chunked<Encoding::kACGT, Layout::kMSB, Checked>(chunked_test_sequences(), extract);
    check_ascii_chunked<Encoding::kACGT, Layout::kLSB, Checked>(chunked_test_sequences(), extract);
    check_ascii_chunked<Encoding::kACTG, Layout::kMSB, Checked>(
        chunked_actg_test_sequences(), extract
    );
    check_ascii_chunked<Encoding::kACTG, Layout::kLSB, Checked>(
        chunked_actg_test_sequences(), extract
    );
    check_ascii_chunked<Encoding::kACGT, Layout::kMSB, Checked>(
        chunked_invalid_sequences(), extract
    );
    check_ascii_chunked<Encoding::kACGT, Layout::kLSB, Checked>(
        chunked_invalid_sequences(), extract
    );
    check_ascii_chunked<Encoding::kACTG, Layout::kMSB, Checked>(
        chunked_invalid_sequences(), extract
    );
    check_ascii_chunked<Encoding::kACTG, Layout::kLSB, Checked>(
        chunked_invalid_sequences(), extract
    );

    using Scratch = std::conditional_t<
        Checked,
        fisk::PackedSequenceWithValidity<Encoding::kACGT, Layout::kMSB>,
        fisk::PackedSequence<Encoding::kACGT, Layout::kMSB>
    >;
    fisk::WordEncoderButterfly<Encoding::kACGT, Layout::kMSB> const encoder;
    Scratch scratch;
    for (Scratch* const s : {static_cast<Scratch*>(nullptr), &scratch}) {
        check_invalid_k_throws(32, [&](std::string const& seq, std::size_t k) {
            std::vector<std::uint64_t> ignored;
            extract.template operator()<7>(
                seq, k, encoder, s,
                ChunkedKmerValueCollector<Encoding::kACGT, Layout::kMSB>{&ignored}
            );
        });
    }
}

// The checked chunked loops as an `extract` for the checks of the extraction from ASCII above,
// called as `extract.template operator()<E, L>(seq, k, callback)`, with a small chunk size, so that
// the sequences of those checks span several chunks, and with the word encoder of E and L.
template <typename Extract>
inline auto ascii_chunked_as_ascii_extractor(Extract const& extract)
{
    return [extract]<fisk::Encoding E, fisk::Layout L>(
        std::string const& seq, std::size_t k, auto const& callback
    ) {
        fisk::PackedSequenceWithValidity<E, L> scratch;
        fisk::WordEncoderButterfly<E, L> const encoder;
        extract.template operator()<7>(seq, k, encoder, &scratch, callback);
    };
}
