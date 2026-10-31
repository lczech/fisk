#pragma once

#include <cstddef>
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

