#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string_view>
#include <vector>

#include "fisk/extractor/kmer_extractor.hpp"
#include "fisk/extractor/spaced_kmer_extractor.hpp"

// The extractors, from ASCII with ACGT and LSB, the fastest encoding and layout, see
// bench_extractor_scaling().
using ScalingKmerChecked = fisk::KmerExtractor<
    fisk::InputFormat::kAscii, fisk::InputCheck::kChecked, fisk::Encoding::kACGT, fisk::Layout::kLSB
>;
using ScalingKmerUnchecked = fisk::KmerExtractor<
    fisk::InputFormat::kAscii, fisk::InputCheck::kUnchecked, fisk::Encoding::kACGT,
    fisk::Layout::kLSB
>;
using ScalingSpacedKmerChecked = fisk::SpacedKmerExtractor<
    fisk::InputFormat::kAscii, fisk::InputCheck::kChecked, fisk::Encoding::kACGT, fisk::Layout::kLSB
>;
using ScalingSpacedKmerUnchecked = fisk::SpacedKmerExtractor<
    fisk::InputFormat::kAscii, fisk::InputCheck::kUnchecked, fisk::Encoding::kACGT,
    fisk::Layout::kLSB
>;

/**
 * @brief Benchmark the extractors over the length of the input, from 64 bases to `max_length`, in
 * steps of powers of two, to see where the caches and the memory set their speed.
 *
 * The extractors are KmerExtractor with k = 31 and SpacedKmerExtractor with the mask W22L31, the
 * common sizes, each checked and unchecked, and each with a callback of one k-mer at a time and
 * with one that also takes the vectors of the SIMD algorithms, see variant.hpp. Each is tuned once
 * for its callback, and measured with the algorithm that tune() chose, which names the benchmark.
 * The input is a random sequence of `max_length` bases, generated as the input of the other
 * suites, see random_acgt(), and each length is a prefix of it, so that it is in memory once.
 */
void bench_extractor_scaling(
    std::size_t max_length,
    double n_prob,
    std::uint64_t seed,
    std::ostream& csv_os
);

// The variants, each defined in its own translation unit (var_*.cpp): the tuning of an extractor
// with the callback of the variant, and its run.
void tune_var_kmer_checked(
    ScalingKmerChecked& extractor,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_kmer_checked(
    ScalingKmerChecked const& extractor,
    std::string_view seq,
    std::vector<std::uint64_t>& sink_buffer
);
void tune_var_kmer_checked_vectors(
    ScalingKmerChecked& extractor,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_kmer_checked_vectors(
    ScalingKmerChecked const& extractor,
    std::string_view seq,
    std::vector<std::uint64_t>& sink_buffer
);
void tune_var_kmer_unchecked(
    ScalingKmerUnchecked& extractor,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_kmer_unchecked(
    ScalingKmerUnchecked const& extractor,
    std::string_view seq,
    std::vector<std::uint64_t>& sink_buffer
);
void tune_var_kmer_unchecked_vectors(
    ScalingKmerUnchecked& extractor,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_kmer_unchecked_vectors(
    ScalingKmerUnchecked const& extractor,
    std::string_view seq,
    std::vector<std::uint64_t>& sink_buffer
);
void tune_var_spaced_kmer_checked(
    ScalingSpacedKmerChecked& extractor,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_spaced_kmer_checked(
    ScalingSpacedKmerChecked const& extractor,
    std::string_view seq,
    std::vector<std::uint64_t>& sink_buffer
);
void tune_var_spaced_kmer_checked_vectors(
    ScalingSpacedKmerChecked& extractor,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_spaced_kmer_checked_vectors(
    ScalingSpacedKmerChecked const& extractor,
    std::string_view seq,
    std::vector<std::uint64_t>& sink_buffer
);
void tune_var_spaced_kmer_unchecked(
    ScalingSpacedKmerUnchecked& extractor,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_spaced_kmer_unchecked(
    ScalingSpacedKmerUnchecked const& extractor,
    std::string_view seq,
    std::vector<std::uint64_t>& sink_buffer
);
void tune_var_spaced_kmer_unchecked_vectors(
    ScalingSpacedKmerUnchecked& extractor,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_spaced_kmer_unchecked_vectors(
    ScalingSpacedKmerUnchecked const& extractor,
    std::string_view seq,
    std::vector<std::uint64_t>& sink_buffer
);
