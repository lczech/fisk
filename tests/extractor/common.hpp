#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "fisk/core/types.hpp"
#include "fisk/extractor/internal.hpp"
#include "fisk/kmer_extract/ascii_chunked.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"
#include "fisk/util/random.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "spaced_kmer_extract/common.hpp"

// =================================================================================================
//     Extractor Test Helpers
// =================================================================================================

// Random sequences, of lengths around the edges of k and of the chunks of the chunked loops, with
// invalid characters when checked.
template <fisk::InputCheck C>
inline std::vector<std::string> extractor_test_sequences()
{
    fisk::Splitmix64 rng(4711);
    std::vector<std::string> seqs;
    std::initializer_list<std::size_t> const lengths = {
        0, 1, 14, 15, 16, 31, 32, 33, 100, 1000, fisk::kDefaultAsciiChunkSize + 77
    };
    for (std::size_t len : lengths) {
        auto seq = random_sequence(rng, len, "ACGT");
        if (C == fisk::InputCheck::kChecked) {
            inject_invalid(seq, rng, 0.01, "N");
        }
        seqs.push_back(std::move(seq));
    }
    return seqs;
}

// The input of an extractor for a sequence: the sequence itself for ASCII, else packed.
template <typename Extractor>
inline auto extractor_test_input(std::string const& seq)
{
    constexpr auto E = Extractor::encoding;
    constexpr auto L = Extractor::layout;
    if constexpr (Extractor::input_format == fisk::InputFormat::kAscii) {
        return seq;
    } else if constexpr (Extractor::input_check == fisk::InputCheck::kChecked) {
        return fisk::pack_sequence_checked(seq, fisk::WordEncoderButterfly<E, L>{});
    } else {
        return fisk::pack_sequence_unchecked(seq, fisk::WordEncoderButterfly<E, L>{});
    }
}

// The k-mers that an extractor is expected to hand out.
template <typename Extractor>
inline std::vector<ExpectedKmer> extractor_test_expected(std::string const& seq, std::size_t k)
{
    constexpr auto E = Extractor::encoding;
    constexpr auto L = Extractor::layout;
    if constexpr (Extractor::input_check == fisk::InputCheck::kChecked) {
        return oracle_kmers_skipping_invalid<E, L>(seq, k);
    } else {
        return oracle_kmers_all_windows<E, L>(seq, k);
    }
}

// A callback of single k-mers that records them.
struct KmerRecorder
{
    std::vector<std::size_t> positions;
    std::vector<std::uint64_t> values;

    template <typename K>
    void operator()(std::size_t pos, K const& kmer)
    {
        positions.push_back(pos);
        values.push_back(fisk::kmer_value(kmer));
    }
};

// A callback of single spaced k-mers that records them.
struct SpacedKmerRecorder
{
    std::vector<SpacedEvent> events;

    template <typename K>
    void operator()(std::size_t pos, std::size_t mask_idx, K const& kmer)
    {
        events.push_back({pos, mask_idx, fisk::kmer_value(kmer)});
    }
};

// The events of each mask in the order of emission, one mask after the other: the order that all
// algorithms guarantee, see SpacedKmerExtractor.
inline std::vector<SpacedEvent> by_mask(std::vector<SpacedEvent> events)
{
    stable_sort_spaced_events(events, [](SpacedEvent const& e) { return e.mask_idx; });
    return events;
}

// Random masks of a span, each with its first and last position kept.
inline std::vector<std::string> random_masks(
    std::size_t count, std::size_t span, std::uint64_t seed
) {
    fisk::Splitmix64 rng(seed);
    std::vector<std::string> masks;
    for (std::size_t i = 0; i < count; ++i) {
        std::string mask(span, '0');
        for (auto& c : mask) {
            c = (rng.get_uint64() & 1u) ? '1' : '0';
        }
        mask.front() = '1';
        mask.back() = '1';
        masks.push_back(std::move(mask));
    }
    return masks;
}
