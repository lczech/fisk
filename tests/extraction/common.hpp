#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "fisk/core/types.hpp"
#include "fisk/extraction/estimate.hpp"
#include "fisk/kmer_extract/ascii_chunked.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"
#include "fisk/util/cpu_runtime.hpp"
#include "fisk/util/random.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "spaced_kmer_extract/common.hpp"

// =================================================================================================
//     Extraction Test Helpers
// =================================================================================================

// Random sequences, of lengths around the edges of k and of the chunks of the chunked loops, with
// invalid characters when checked.
template <fisk::InputCheck C>
inline std::vector<std::string> extraction_test_sequences()
{
    fisk::Splitmix64 rng(4711);
    std::vector<std::string> seqs;
    std::initializer_list<std::size_t> const lengths = {
        0, 1, 14, 15, 16, 31, 32, 33, 56, 57, 58, 63, 64, 65, 100, 1000,
        fisk::kDefaultAsciiChunkSize + 77
    };
    for (std::size_t len : lengths) {
        auto seq = random_sequence(rng, len, "ACGT");
        if (C == fisk::InputCheck::kChecked) {
            inject_invalid(seq, rng, 0.01, "N");
        }
        seqs.push_back(std::move(seq));
    }

    // A single invalid character, which for the sparse masks of extraction_wide_mask_sets() is at
    // skipped positions of most windows that contain it.
    if (C == fisk::InputCheck::kChecked) {
        auto seq = random_sequence(rng, 130, "ACGT");
        seq[70] = 'N';
        seqs.push_back(std::move(seq));
    }
    return seqs;
}

// The input of the extraction for a sequence: the sequence itself for ASCII, else packed.
template <fisk::InputFormat I, fisk::InputCheck C, fisk::Encoding E, fisk::Layout L>
inline auto extraction_test_input(std::string const& seq)
{
    if constexpr (I == fisk::InputFormat::kAscii) {
        return seq;
    } else if constexpr (C == fisk::InputCheck::kChecked) {
        return fisk::pack_sequence_checked(seq, fisk::WordEncoderButterfly<E, L>{});
    } else {
        return fisk::pack_sequence_unchecked(seq, fisk::WordEncoderButterfly<E, L>{});
    }
}

// The k-mers that the extraction is expected to hand out.
template <fisk::InputCheck C, fisk::Encoding E, fisk::Layout L>
inline std::vector<ExpectedKmer> extraction_test_expected(std::string const& seq, std::size_t k)
{
    if constexpr (C == fisk::InputCheck::kChecked) {
        return oracle_kmers_skipping_invalid<E, L>(seq, k);
    } else {
        return oracle_kmers_all_windows<E, L>(seq, k);
    }
}

// The same, for an extractor, such as KmerExtractor, by its choices.
template <typename Extractor>
inline auto extraction_test_input(std::string const& seq)
{
    return extraction_test_input<
        Extractor::input_format, Extractor::input_check, Extractor::encoding, Extractor::layout
    >(seq);
}

template <typename Extractor>
inline std::vector<ExpectedKmer> extraction_test_expected(std::string const& seq, std::size_t k)
{
    return extraction_test_expected<Extractor::input_check, Extractor::encoding, Extractor::layout>(
        seq, k
    );
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
// algorithms guarantee, see for_each_spaced_kmer().
inline std::vector<SpacedEvent> by_mask(std::vector<SpacedEvent> events)
{
    stable_sort_spaced_events(events, [](SpacedEvent const& e) { return e.mask_idx; });
    return events;
}

// Random masks of a span, each with its first and last position kept. Up to span 32, each position
// is kept with a chance of one half; above, at most 32 random positions are kept.
inline std::vector<std::string> random_masks(
    std::size_t count, std::size_t span, std::uint64_t seed
) {
    fisk::Splitmix64 rng(seed);
    std::vector<std::string> masks;
    for (std::size_t i = 0; i < count; ++i) {
        std::string mask(span, '0');
        if (span <= 32) {
            for (auto& c : mask) {
                c = (rng.get_uint64() & 1u) ? '1' : '0';
            }
        } else {
            std::size_t const weight = 2 + rng.get_uint64() % 29;
            for (std::size_t j = 0; j < weight; ++j) {
                mask[rng.get_uint64() % span] = '1';
            }
        }
        mask.front() = '1';
        mask.back() = '1';
        masks.push_back(std::move(mask));
    }
    return masks;
}

// Sets of masks with spans around the boundaries of the pieces, see SpacedKmerMask: one piece up to
// span 32, two up to 57, and three from 58 on, which the extraction loops run in their loop over
// the pieces at runtime, see kDefaultMaxCompiledPieces. Each set has kMaxTestMasks masks, and the
// last one is a single mask of span 64 with no kept position in its middle piece.
inline std::vector<std::vector<std::string>> extraction_wide_mask_sets()
{
    std::vector<std::vector<std::string>> sets;
    for (std::size_t span : {32u, 33u, 57u, 58u, 64u}) {
        sets.push_back(random_masks(kMaxTestMasks, span, span));
    }
    sets.push_back({ std::string("1").append(62, '0').append("1") });
    return sets;
}

// The SIMD arch of the estimates, see CandidateSimdArchs (extraction/estimate.hpp): the widest
// one compiled of AVX2, Neon, and SSE2, where the CPU has it, else empty, as in the names of the
// algorithms.
inline std::string estimate_simd_arch_name()
{
    #if defined(FISK_HAS_AVX2)
    return fisk::avx2_enabled() ? "avx2" : "";
    #elif defined(FISK_HAS_NEON)
    return fisk::neon_enabled() ? "neon" : "";
    #elif defined(FISK_HAS_SSE2)
    return fisk::sse2_enabled() ? "sse2" : "";
    #else
    return "";
    #endif
}

// The estimate of for_each_kmer(): the SIMD loop of the arch of estimate_simd_arch_name(), else the
// scalar loop; `loop` is the name of the function of the loop for the input, such as
// "for_each_kmer_ascii_checked_chunked".
inline std::string expected_kmer_estimate(std::string const& loop)
{
    auto const arch = estimate_simd_arch_name();
    return loop + (arch.empty() ? "_scalar" : "_simd_" + arch);
}

// The estimate of for_each_spaced_kmer(): PEXT where likely fast, else the butterfly in the SIMD
// loop of the arch of estimate_simd_arch_name(), else in the scalar loop; `loop` as above.
inline std::string expected_spaced_estimate(std::string const& loop)
{
    if (fisk::pext_likely_fast()) {
        return loop + "_scalar_pext";
    }
    auto const arch = estimate_simd_arch_name();
    return loop + (arch.empty() ? "_scalar_butterfly" : "_simd_butterfly_" + arch);
}
