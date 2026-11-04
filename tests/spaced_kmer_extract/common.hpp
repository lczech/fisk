#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/core/cast.hpp"
#include "fisk/core/types.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"
#include "fisk/util/random.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "testing.hpp"

// =================================================================================================
//     Helpers
// =================================================================================================

// The sets of masks of the heavier spaced k-mer tests, which check many bit extractors and callback
// forms, have at most this many masks. They dispatch them with
// dispatch_static_span<kMaxTestMasks>(), which compiles the loops for the sizes up to this only,
// rather than for every size up to kDefaultMaxMasks, and throws for a larger set. Larger sets
// are checked on their own, with few extractors: up to kDefaultMaxMasks in
// packed_scalar_common.hpp, and up to more than that in test_ascii_rolling_simd.cpp.
inline constexpr std::size_t kMaxTestMasks = 3;

// Maximum span of the masks of the tests of spans up to 32, see SpacedKmerMask: masks that can
// have more pieces would also compile the loops for them, which these tests do not run, see the
// tests of spans above 32 for those.
inline constexpr std::size_t kNarrowMaxSpan = 32;

// The sequences of the tests of the extraction from packed sequences, scalar and SIMD.
//
// Sequence lengths 0..100, which covers sequences shorter than the span, the last step, and the
// main loop for every span and arch, plus two longer ones, each with random per-position content.
// The oracle (oracle.hpp) works on the ASCII input, independent of the packing and the extraction.
inline std::vector<std::string> const& packed_test_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        fisk::Splitmix64 rng(6006);
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
inline std::vector<std::string> const& packed_checked_test_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        fisk::Splitmix64 rng(8008);
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

// For the spans of one piece at a maximum span above 32, and for spans above 32, around the
// boundaries of their pieces, see SpacedKmerMask, a single mask and a set of three masks, each with
// random kept positions, at most 32 of them. Plus single masks that keep only the first and the
// last position, so that the pieces between them keep none, and masks that keep 32 positions
// spread over all pieces.
inline std::vector<std::vector<std::string>> const& wide_test_mask_sets()
{
    static std::vector<std::vector<std::string>> const sets = [] {
        fisk::Splitmix64 rng(7009);
        auto random_mask = [&](std::size_t span_k) {
            std::string mask(span_k, '0');
            std::size_t const weight = 2 + rng.get_uint64() % 31;
            for (std::size_t i = 0; i < weight; ++i) {
                mask[rng.get_uint64() % span_k] = '1';
            }
            mask.front() = '1';
            mask.back() = '1';
            return mask;
        };
        std::vector<std::vector<std::string>> out;
        for (std::size_t span_k : {
            29u, 30u, 31u, 32u, 33u, 34u, 56u, 57u, 58u, 60u, 64u, 84u, 85u, 86u, 100u, 113u,
            114u, 128u
        }) {
            out.push_back({ random_mask(span_k) });
            out.push_back({ random_mask(span_k), random_mask(span_k), random_mask(span_k) });
        }
        for (std::size_t span_k : {60u, 86u, 128u}) {
            out.push_back({ std::string("1").append(span_k - 2, '0').append("1") });
        }
        for (std::size_t span_k : {64u, 100u, 128u}) {
            std::string mask(span_k, '0');
            for (std::size_t i = 0; i < 32; ++i) {
                mask[i * (span_k - 1) / 31] = '1';
            }
            out.push_back({ mask });
        }
        return out;
    }();
    return sets;
}

// Sequences around the span and the SIMD block width: empty, one short of a span, and a span plus
// up to two blocks, with random content (rather than a periodic pattern, under which a k-mer taken
// from the wrong window could still have the right value). Then invalid characters: hand-picked
// ones at the start, around the end of the first window, and in runs, and random bytes from the
// full range at two densities.
inline std::vector<std::string> sequence_corpus(std::size_t span_k, std::size_t lanes)
{
    fisk::Splitmix64 rng(1000 * span_k + lanes);
    std::size_t marker_idx = 0;
    auto marker = [&] { return kHandPickedInvalid[marker_idx++ % kHandPickedInvalid.size()]; };

    std::vector<std::string> sequences;
    sequences.push_back("");
    if (span_k > 1) {
        sequences.push_back(random_sequence(rng, span_k - 1));
    }
    for (std::size_t extra = 0; extra <= 2 * lanes; ++extra) {
        sequences.push_back(random_sequence(rng, span_k + extra));
    }

    std::string invalid = random_sequence(rng, span_k + 2 * lanes + 5);
    invalid[0] = marker();
    if (span_k > 2) {
        invalid[1] = marker();
    }
    invalid[span_k - 1] = marker();
    invalid[span_k] = marker();
    sequences.push_back(invalid);

    // N at index 1 is skipped by 10101, while N at index 2 is kept and suppresses position 0.
    for (std::string seq : {"ANCGTACGTACGT", "ACNGTACGTACGT", "ACGTNNNACGTACGT"}) {
        std::replace(seq.begin(), seq.end(), 'N', marker());
        sequences.push_back(seq);
    }

    for (double const rate : {0.05, 0.2}) {
        for (std::size_t i = 0; i < 4; ++i) {
            std::string seq = random_sequence(rng, span_k + 3 * lanes + i);
            inject_invalid(seq, rng, rate, all_invalid_bytes());
            sequences.push_back(std::move(seq));
        }
    }
    return sequences;
}

// =================================================================================================
//     Chunked Extraction from ASCII
// =================================================================================================

// The chunked loops pack the input chunk by chunk, see kmer_extract/ascii_chunked.hpp. They are
// checked through a callable `extract`, called as
// `extract.template operator()<ChunkSize, E, L>(seq, mask_strings, single, scratch, callback)`,
// which runs the loop with the masks or kernels of `mask_strings`, a set of one or three masks of
// the same span: with `single` a std::true_type, through the overload for a single mask, and with
// a std::false_type through the overload for a std::span of fixed size. `scratch` points to the
// scratch to use, or is null for the overload with the thread-local buffer. `callback` takes single
// spaced k-mers, in all forms; the loops that call back with vectors hand on their lanes with
// for_each_valid_lane(). The oracle decodes directly from the ASCII input, since these checks are
// also meant to exercise the packing.

// The callbacks of the checks below, in all forms, as named types rather than lambdas: the
// lambdas of a check would have a type of their own for each chunk size, which would compile the
// packed loops once per chunk size.
template <fisk::Encoding E, fisk::Layout L>
struct ChunkedSpacedEventCollector
{
    std::vector<SpacedEvent>* events;

    void operator()(
        std::size_t pos, std::size_t mask_idx, fisk::SpacedKmer<E, L> const& spaced_kmer
    ) const {
        events->push_back({pos, mask_idx, fisk::kmer_value(spaced_kmer)});
    }
};

template <fisk::Encoding E, fisk::Layout L>
struct ChunkedSpacedPosCollector
{
    std::vector<SpacedEvent>* events;

    void operator()(std::size_t pos, fisk::SpacedKmer<E, L> const& spaced_kmer) const
    {
        events->push_back({pos, 0, fisk::kmer_value(spaced_kmer)});
    }
};

template <fisk::Encoding E, fisk::Layout L>
struct ChunkedSpacedValueCollector
{
    std::vector<SpacedEvent>* events;

    void operator()(fisk::SpacedKmer<E, L> const& spaced_kmer) const
    {
        events->push_back({0, 0, fisk::kmer_value(spaced_kmer)});
    }
};

// For the spans 1, 2, 7, 15, and 29 to 32 (the full-width loops), a single random mask and a set
// of three random masks, which keep the first and the last position, plus a mask that keeps all
// positions, and one that keeps only the first and the last.
inline std::vector<std::vector<std::string>> const& chunked_mask_sets()
{
    static std::vector<std::vector<std::string>> const sets = [] {
        fisk::Splitmix64 rng(7117);
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
        for (std::size_t span_k : {1u, 2u, 7u, 15u, 29u, 30u, 31u, 32u}) {
            out.push_back({ random_mask(span_k) });
            out.push_back({ random_mask(span_k), random_mask(span_k), random_mask(span_k) });
        }
        out.push_back({ std::string(32, '1') });
        out.push_back({ std::string("1").append(30, '0').append("1") });
        return out;
    }();
    return sets;
}

// For spans above 32, around the boundaries of the pieces of the masks, see SpacedKmerMask, and the
// span 128, with five pieces: a single random mask and a set of three, which keep the first and the
// last position, and at most 32 positions.
inline std::vector<std::vector<std::string>> const& chunked_wide_mask_sets()
{
    static std::vector<std::vector<std::string>> const sets = [] {
        fisk::Splitmix64 rng(7118);
        auto random_mask = [&](std::size_t span_k) {
            std::string mask(span_k, '0');
            std::size_t const weight = 2 + rng.get_uint64() % 31;
            for (std::size_t i = 0; i < weight; ++i) {
                mask[rng.get_uint64() % span_k] = '1';
            }
            mask.front() = '1';
            mask.back() = '1';
            return mask;
        };
        std::vector<std::vector<std::string>> out;
        for (std::size_t span_k : {33u, 57u, 58u, 85u, 100u, 128u}) {
            out.push_back({ random_mask(span_k) });
            out.push_back({ random_mask(span_k), random_mask(span_k), random_mask(span_k) });
        }
        return out;
    }();
    return sets;
}

// Sequences of the lengths 0..80 and two longer ones, with random nucleotides, and with checked,
// also with invalid characters: the corpus of the span, see sequence_corpus(), and the same lengths
// with scattered invalid characters, and with many.
inline std::vector<std::string> chunked_spaced_sequences(std::size_t span_k, bool checked)
{
    fisk::Splitmix64 rng(5115 + span_k);
    std::vector<std::string> out;
    for (std::size_t len = 0; len <= 80; ++len) {
        std::string seq = random_sequence(rng, len);
        if (checked) {
            double const rates[] = {0.0, 0.05, 0.2};
            inject_invalid(seq, rng, rates[len % 3], all_invalid_bytes());
        }
        out.push_back(std::move(seq));
    }
    for (std::size_t len : {150u, 257u}) {
        std::string seq = random_sequence(rng, len);
        if (checked) {
            inject_invalid(seq, rng, 0.02, kHandPickedInvalid);
            seq.replace(len / 2, 40, 40, 'N');
        }
        out.push_back(std::move(seq));
    }
    if (checked) {
        for (auto& seq : sequence_corpus(span_k, 8)) {
            out.push_back(std::move(seq));
        }
    }
    return out;
}

// Calls `f(std::integral_constant<std::size_t, ChunkSize>{})` for chunk sizes smaller than most
// spans, around the blocks of 32 positions of the packed loops, and larger than the longest test
// sequence. Few of them, as the chunked loops are inlined into each chunk size, which makes them
// expensive to compile. See for_each_chunk_size() (kmer_extract/common.hpp) for why the size is a
// value.
template <typename F>
inline void for_each_spaced_chunk_size(F&& f)
{
    [&]<std::size_t... Sizes>(std::index_sequence<Sizes...>) {
        (f(std::integral_constant<std::size_t, Sizes>{}), ...);
    }(std::index_sequence<1, 7, 32, 33, 1000>{});
}

// Sorts `events` by `key`, such as their position or mask, keeping their order for equal keys.
// Not with std::stable_sort(), whose temporary buffer is deprecated in some libstdc++ versions;
// instead, the index of each event breaks the ties.
template <typename Key>
inline void stable_sort_spaced_events(std::vector<SpacedEvent>& events, Key key)
{
    std::vector<std::size_t> order(events.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        auto const key_a = key(events[a]);
        auto const key_b = key(events[b]);
        return key_a != key_b ? key_a < key_b : a < b;
    });
    std::vector<SpacedEvent> sorted;
    sorted.reserve(events.size());
    for (std::size_t const i : order) {
        sorted.push_back(events[i]);
    }
    events = std::move(sorted);
}

// Runs `extract` with the given chunk size and conventions into `events`, which `callback` fills,
// and with `ByPosition`, sorts them by position, see check_spaced_ascii_chunked(). A function
// rather than a lambda in there, as MSVC fails on a lambda that uses the chunk size of its
// enclosing one.
template <
    std::size_t ChunkSize, fisk::Encoding E, fisk::Layout L, bool ByPosition, typename Extract,
    typename Single, typename Scratch, typename Callback
>
inline void run_spaced_ascii_chunked_(
    Extract const& extract, std::string const& seq, std::vector<std::string> const& mask_strings,
    Single single, Scratch* scratch, Callback const& callback, std::vector<SpacedEvent>& events
) {
    events.clear();
    extract.template operator()<ChunkSize, E, L>(seq, mask_strings, single, scratch, callback);
    if constexpr (ByPosition) {
        stable_sort_spaced_events(events, [](SpacedEvent const& e) { return e.pos; });
    }
}

// Checks `extract` with the word encoder of the given conventions against the oracle, for every
// set of masks of `mask_sets`, sequence, and chunk size, with a scratch and with the thread-local
// buffer: through the overload for a span of masks, and for a single mask also through the
// overload for a single mask, in all its callback forms; the form without positions takes them
// from the first form, so that it checks the values and their order. Unchecked, on sequences
// without invalid characters. With `ByPosition`, the events are sorted by position first, keeping
// the order of the masks at each position, for the loops that call back with vectors of positions
// per mask.
template <fisk::Encoding E, fisk::Layout L, bool Checked, bool ByPosition, typename Extract>
inline void check_spaced_ascii_chunked(
    Extract const& extract, std::vector<std::vector<std::string>> const& mask_sets
) {
    using Scratch = std::conditional_t<
        Checked, fisk::PackedSequenceWithValidity<E, L>, fisk::PackedSequence<E, L>
    >;
    Scratch scratch;
    std::vector<SpacedEvent> events;
    std::vector<SpacedEvent> first;
    for (auto const& mask_strings : mask_sets) {
        std::size_t const span_k = mask_strings.front().size();
        for (auto const& seq : chunked_spaced_sequences(span_k, Checked)) {
            auto const expected = oracle_spaced_events<E, L>(seq, mask_strings);
            for_each_spaced_chunk_size([&](auto chunk_size) {
                constexpr std::size_t ChunkSize = decltype(chunk_size)::value;
                for (Scratch* const s : {static_cast<Scratch*>(nullptr), &scratch}) {
                    run_spaced_ascii_chunked_<ChunkSize, E, L, ByPosition>(
                        extract, seq, mask_strings, std::false_type{}, s,
                        ChunkedSpacedEventCollector<E, L>{&events}, events
                    );
                    check_spaced_events(events, expected);
                    if (mask_strings.size() != 1) {
                        continue;
                    }
                    run_spaced_ascii_chunked_<ChunkSize, E, L, ByPosition>(
                        extract, seq, mask_strings, std::true_type{}, s,
                        ChunkedSpacedEventCollector<E, L>{&events}, events
                    );
                    check_spaced_events(events, expected);
                    run_spaced_ascii_chunked_<ChunkSize, E, L, ByPosition>(
                        extract, seq, mask_strings, std::true_type{}, s,
                        ChunkedSpacedPosCollector<E, L>{&events}, events
                    );
                    check_spaced_events(events, expected);
                    first = events;
                    run_spaced_ascii_chunked_<ChunkSize, E, L, ByPosition>(
                        extract, seq, mask_strings, std::true_type{}, s,
                        ChunkedSpacedValueCollector<E, L>{&events}, events
                    );
                    for (std::size_t i = 0; i < std::min(first.size(), events.size()); ++i) {
                        events[i].pos = first[i].pos;
                    }
                    check_spaced_events(events, expected);
                }
            });
        }
    }
}

// The checks above under the four conventions, by default with the mask sets of
// chunked_mask_sets().
template <bool Checked, bool ByPosition, typename Extract>
inline void check_spaced_ascii_chunked_all(
    Extract const& extract,
    std::vector<std::vector<std::string>> const& mask_sets = chunked_mask_sets()
) {
    using fisk::Encoding;
    using fisk::Layout;
    check_spaced_ascii_chunked<Encoding::kACGT, Layout::kMSB, Checked, ByPosition>(
        extract, mask_sets
    );
    check_spaced_ascii_chunked<Encoding::kACGT, Layout::kLSB, Checked, ByPosition>(
        extract, mask_sets
    );
    check_spaced_ascii_chunked<Encoding::kACTG, Layout::kMSB, Checked, ByPosition>(
        extract, mask_sets
    );
    check_spaced_ascii_chunked<Encoding::kACTG, Layout::kLSB, Checked, ByPosition>(
        extract, mask_sets
    );
}

// Checks that the chunked loops reject spans outside of [1, MaxSpan] and sets of masks with
// different spans, through the overload for a single mask and for a set of masks, also for
// sequences shorter than the span; with spans above 32 among the different ones. The masks are
// those of BitExtractMask in kMSB, as `convert(masks)` turns a std::vector of them into a
// std::vector of the masks or kernels of the loops. `expect_throw(seq, mask_or_masks)` runs the
// loops on them, with a scratch and with the thread-local buffer, and expects each to throw
// std::invalid_argument.
template <std::size_t MaxSpan, typename Convert, typename ExpectThrow>
inline void check_spaced_ascii_chunked_error_contract(
    Convert const& convert, ExpectThrow const& expect_throw
) {
    constexpr fisk::Layout L = fisk::Layout::kMSB;
    using SourceMask = fisk::SpacedKmerMask<L, fisk::BitExtractMask, MaxSpan>;
    for (std::string const& seq : {std::string{}, std::string(2 * MaxSpan + 1, 'T')}) {
        // A mask with an invalid span, which make_spaced_kmer_mask() would reject.
        for (std::size_t const span_k : {std::size_t{0}, MaxSpan + 1}) {
            SourceMask source;
            source.mask_pieces[0] = fisk::BitExtractMask(3);
            source.span_k = static_cast<std::uint16_t>(span_k);
            auto const masks = convert(std::vector<SourceMask>{source});
            using Mask = typename std::remove_cvref_t<decltype(masks)>::value_type;
            expect_throw(seq, masks.front());
            expect_throw(seq, std::span<Mask const, 1>(masks.data(), 1));
        }
        std::string const span_40 = std::string("1").append(38, '0').append("1");
        std::string const span_41 = std::string("1").append(39, '0').append("1");
        for (auto const& mask_strings : std::vector<std::vector<std::string>>{
            {"101", "1001"}, {span_40, span_41}
        }) {
            auto const mixed = convert(fisk::make_spaced_kmer_masks<L, MaxSpan>(mask_strings));
            using Mask = typename std::remove_cvref_t<decltype(mixed)>::value_type;
            expect_throw(seq, std::span<Mask const, 2>(mixed.data(), 2));
        }
    }
}
