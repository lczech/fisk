#pragma once

#include <chrono>
#include <cstddef>

namespace fisk {

// =================================================================================================
//     Tune Options
// =================================================================================================

/**
 * @brief The largest number of rounds of the in-situ tuning, see TuneOptions::rounds.
 */
inline constexpr std::size_t kTuneMaxRounds = std::size_t{1} << 16;

/**
 * @brief The largest number of k-mers of the pieces of the in-situ tuning, see
 * TuneOptions::piece_length.
 */
inline constexpr std::size_t kTuneMaxPieceLength = std::size_t{1} << 24;

/**
 * @brief Options of tune(), see KmerExtractor and SpacedKmerExtractor: the in-situ tuning, in the
 * calls of for_each() of the extractor, on their sequences and with their callbacks.
 */
struct TuneOptions
{
    /**
     * @brief The time that each algorithm runs in the calls of for_each() while tuning, split
     * into `rounds` turns.
     *
     * The turns of the algorithms alternate on the sequences of those calls, so that the first
     * `time_per_algorithm` times the number of algorithms of for_each() are spent tuning. Longer
     * times give more stable choices between algorithms of about the same speed; the default is
     * long enough for stable choices, and longer ones mostly spend more input on the slower
     * algorithms. With 0 or less, each turn runs one piece, see `piece_length`, or one shorter
     * sequence, so that the tuning takes a fixed amount of input, whatever the speed of the
     * machine.
     */
    std::chrono::nanoseconds time_per_algorithm = std::chrono::milliseconds(5);

    /**
     * @brief The number of turns of each algorithm while tuning in situ, which alternate with
     * those of the others; its fastest turn counts. 0 counts as 1, and more than kTuneMaxRounds
     * as kTuneMaxRounds.
     */
    std::size_t rounds = 3;

    /**
     * @brief The number of k-mers of the pieces that longer sequences are cut into while tuning in
     * situ, so that the turns of the algorithms can alternate within a sequence. A turn reads the
     * clock after each piece, or after as many k-mers of shorter sequences. Rounded up to a
     * multiple of 256, at least 256, so that 0 counts as 256, and at most kTuneMaxPieceLength, so
     * that a turn reads the clock at least that often; rarely needs to be changed.
     */
    std::size_t piece_length = std::size_t{1} << 16;
};

} // namespace fisk
