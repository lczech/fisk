#pragma once

#include <concepts>
#include <cstddef>
#include <ranges>

namespace fisk {

// =================================================================================================
//     Tune Options
// =================================================================================================

/**
 * @brief How much effort tune() spends on choosing the algorithm of an extractor, see KmerExtractor
 * and SpacedKmerExtractor.
 *
 * `kEstimate`: no timing; a fixed order of preference, by what is compiled and available on the
 * CPU. This is also the choice of a newly constructed extractor.
 *
 * `kMeasure`: time a curated list of the algorithms that are fast on some hardware, and use the
 * fastest of them.
 */
enum class TuneLevel
{
    kEstimate,
    kMeasure
};

/**
 * @brief Options of tune(), see KmerExtractor and SpacedKmerExtractor.
 */
struct TuneOptions
{
    /**
     * @brief Estimate or measure, see TuneLevel.
     */
    TuneLevel level = TuneLevel::kMeasure;

    /**
     * @brief The number of random bases to time the algorithms on, when no sample is given.
     */
    std::size_t total_length = std::size_t{1} << 20;

    /**
     * @brief The length of the random sequences that `total_length` is split into, for instance the
     * length of short reads, or 0 for one sequence of `total_length`.
     */
    std::size_t sequence_length = 0;

    /**
     * @brief The rate of invalid characters (`N`) in the random sequences of the checked
     * extraction, so that its loops leave their paths for all-valid input as on real data.
     */
    double invalid_rate = 0.001;

    /**
     * @brief The time of each timed run of an algorithm, in seconds: its number of passes over the
     * sample, at least one, is set so that a run takes this long.
     */
    double sample_seconds = 0.002;

    /**
     * @brief The number of timed runs of each algorithm, which take turns with those of the
     * others; the fastest counts.
     */
    std::size_t repeats = 5;
};

// =================================================================================================
//     Tune Samples
// =================================================================================================

/**
 * @brief A sample for tune(): one sequence that converts to `Seq`, or a range of them.
 */
template <typename T, typename Seq>
concept ExtractorSample =
    std::convertible_to<T const&, Seq const&> || (
        std::ranges::input_range<T const> &&
        std::convertible_to<std::ranges::range_reference_t<T const>, Seq const&>
    );

} // namespace fisk
