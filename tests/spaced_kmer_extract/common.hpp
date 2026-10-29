#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "fisk/core/cast.hpp"
#include "fisk/core/types.hpp"
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
// rather than for every size up to kMaxDispatchedMasks, and throws for a larger set. Larger sets,
// up to kMaxDispatchedMasks, are checked on their own, with few extractors, see
// test_packed_bytewise_scalar.cpp and test_ascii_rolling_simd.cpp.
inline constexpr std::size_t kMaxTestMasks = 3;

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

// Sorts `events` by position, keeping their order at each position, such as that of the masks.
// Not with std::stable_sort(), whose temporary buffer is deprecated in some libstdc++ versions;
// instead, the index of each event breaks the ties.
inline void sort_spaced_events_by_position(std::vector<SpacedEvent>& events)
{
    std::vector<std::size_t> order(events.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return events[a].pos != events[b].pos ? events[a].pos < events[b].pos : a < b;
    });
    std::vector<SpacedEvent> sorted;
    sorted.reserve(events.size());
    for (std::size_t const i : order) {
        sorted.push_back(events[i]);
    }
    events = std::move(sorted);
}

// Checks `extract` with the word encoder of the given conventions against the oracle, for every
// set of masks, sequence, and chunk size, with a scratch and with the thread-local buffer: through
// the overload for a span of masks, and for a single mask also through the overload for a single
// mask, in all its callback forms; the form without positions takes them from the first form, so
// that it checks the values and their order. Unchecked, on sequences without invalid characters.
// With `ByPosition`, the events are sorted by position first, keeping the order of the masks at
// each position, for the loops that call back with vectors of positions per mask.
template <fisk::Encoding E, fisk::Layout L, bool Checked, bool ByPosition, typename Extract>
inline void check_spaced_ascii_chunked(Extract const& extract)
{
    using Scratch = std::conditional_t<
        Checked, fisk::PackedSequenceWithValidity<E, L>, fisk::PackedSequence<E, L>
    >;
    Scratch scratch;
    std::vector<SpacedEvent> events;
    std::vector<SpacedEvent> first;
    for (auto const& mask_strings : chunked_mask_sets()) {
        std::size_t const span_k = mask_strings.front().size();
        for (auto const& seq : chunked_spaced_sequences(span_k, Checked)) {
            auto const expected = oracle_spaced_events<E, L>(seq, mask_strings);
            for_each_spaced_chunk_size([&](auto chunk_size) {
                constexpr std::size_t ChunkSize = decltype(chunk_size)::value;
                for (Scratch* const s : {static_cast<Scratch*>(nullptr), &scratch}) {
                    auto const run = [&](auto single, auto const& callback) {
                        events.clear();
                        extract.template operator()<ChunkSize, E, L>(
                            seq, mask_strings, single, s, callback
                        );
                        if constexpr (ByPosition) {
                            sort_spaced_events_by_position(events);
                        }
                    };
                    run(std::false_type{}, ChunkedSpacedEventCollector<E, L>{&events});
                    check_spaced_events(events, expected);
                    if (mask_strings.size() != 1) {
                        continue;
                    }
                    run(std::true_type{}, ChunkedSpacedEventCollector<E, L>{&events});
                    check_spaced_events(events, expected);
                    run(std::true_type{}, ChunkedSpacedPosCollector<E, L>{&events});
                    check_spaced_events(events, expected);
                    first = events;
                    run(std::true_type{}, ChunkedSpacedValueCollector<E, L>{&events});
                    for (std::size_t i = 0; i < std::min(first.size(), events.size()); ++i) {
                        events[i].pos = first[i].pos;
                    }
                    check_spaced_events(events, expected);
                }
            });
        }
    }
}

// The checks above under the four conventions.
template <bool Checked, bool ByPosition, typename Extract>
inline void check_spaced_ascii_chunked_all(Extract const& extract)
{
    using fisk::Encoding;
    using fisk::Layout;
    check_spaced_ascii_chunked<Encoding::kACGT, Layout::kMSB, Checked, ByPosition>(extract);
    check_spaced_ascii_chunked<Encoding::kACGT, Layout::kLSB, Checked, ByPosition>(extract);
    check_spaced_ascii_chunked<Encoding::kACTG, Layout::kMSB, Checked, ByPosition>(extract);
    check_spaced_ascii_chunked<Encoding::kACTG, Layout::kLSB, Checked, ByPosition>(extract);
}
