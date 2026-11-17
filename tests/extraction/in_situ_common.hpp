#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "fisk/extraction/tune_options.hpp"
#include "fisk/extraction/extractor_plan.hpp"
#include "fisk/util/random.hpp"
#include "corpus.hpp"

// =================================================================================================
//     In-Situ Tuning Test Helpers
// =================================================================================================

// The tests of the in-situ tuning, see ExtractorPlan::tune(), tune with turns of one piece
// of kInSituTestPiece k-mers each (a time of 0), so that the tuning ends after a fixed number of
// pieces, and compare the k-mers of the calls of for_each() during and after it with those of the
// oracle.
inline constexpr std::size_t kInSituTestPiece = fisk::kExtractorPieceUnit;

// The options of the in-situ tuning of the tests.
inline fisk::TuneOptions in_situ_test_options(std::size_t rounds)
{
    fisk::TuneOptions options;
    options.time_per_algorithm = std::chrono::nanoseconds(0);
    options.rounds = rounds;
    options.piece_length = kInSituTestPiece;
    return options;
}

// The lengths of the long sequences of the tests: around the boundaries of the pieces and of the
// overlap of their k-mers, and random ones up to 5000.
inline std::vector<std::size_t> in_situ_test_lengths(std::size_t overlap, fisk::Splitmix64& rng)
{
    constexpr std::size_t p = kInSituTestPiece;
    std::size_t const k = overlap + 1;
    std::vector<std::size_t> lengths = {
        0, 1, k - 1, k, k + 1, p - 1, p, p + 1, p + k - 2, p + k - 1, p + k,
        2 * p - 1, 2 * p, 2 * p + 1, 2 * p + k - 1, 3 * p + k - 1, 3 * p + k, 3 * p + k + 1,
    };
    for (std::size_t i = 0; i < 200; ++i) {
        lengths.push_back(rng.get_uint64() % 5001);
    }
    return lengths;
}

// The input of a test: `prefix` short sequences, each with up to one piece of k-mers, so that the
// tuning ends at another piece of the long sequence after them, and then the long sequence of
// `length` nucleotides; with invalid characters when checked.
template <fisk::InputCheck C>
inline std::vector<std::string> in_situ_test_input(
    std::size_t prefix, std::size_t length, fisk::Splitmix64& rng
) {
    static constexpr std::size_t short_lengths[] = {150, 0, 40, 255, 3};
    std::vector<std::string> seqs;
    for (std::size_t i = 0; i < prefix; ++i) {
        seqs.push_back(random_sequence(rng, short_lengths[i % 5], "ACGT"));
    }
    seqs.push_back(random_sequence(rng, length, "ACGT"));
    if constexpr (C == fisk::InputCheck::kChecked) {
        for (auto& seq : seqs) {
            inject_invalid(seq, rng, 0.01, "N");
        }
    }
    return seqs;
}
