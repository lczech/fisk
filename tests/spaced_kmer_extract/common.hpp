#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "fisk/core/random.hpp"
#include "corpus.hpp"

// =================================================================================================
//     Helpers
// =================================================================================================

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
