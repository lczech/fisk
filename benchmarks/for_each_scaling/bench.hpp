#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string_view>

#include "fisk/extraction/spaced_kmer_mask_forms.hpp"
#include "sink.hpp"

// The k of the k-mers, and the masks of the spaced k-mers, see bench_for_each_scaling(). From
// ASCII with ACGT and LSB, the fastest encoding and layout.
inline constexpr std::size_t kScalingK = 31;
using ScalingMaskForms = fisk::SpacedKmerMaskForms<fisk::Layout::kLSB>;

/**
 * @brief Benchmark for_each_kmer() and for_each_spaced_kmer() over the length of the input, from 64
 * bases to `max_length`, in steps of powers of two, to see where the caches and the memory set
 * their speed, and what a call costs on short input.
 *
 * The k-mers have k = 31, and the spaced k-mers the mask W22L31, the common sizes, each checked and
 * unchecked, and each with a callback of one k-mer at a time and with one that also takes the
 * vectors of the SIMD algorithms, see variant.hpp. Each runs the algorithm of the estimate, which
 * names the benchmark. The input is a random sequence of `max_length` bases, generated as the
 * input of the other suites, see random_acgt(), and each length is a prefix of it, so that it is in
 * memory once.
 */
void bench_for_each_scaling(
    std::size_t max_length,
    double n_prob,
    std::uint64_t seed,
    std::ostream& csv_os
);

// The variants, each defined in its own translation unit (var_*.cpp): the extraction with the
// callback of the variant.
std::uint64_t run_var_kmer_checked(std::string_view seq, SinkBuffer& sink_buffer);
std::uint64_t run_var_kmer_checked_vectors(std::string_view seq, SinkBuffer& sink_buffer);
std::uint64_t run_var_kmer_unchecked(std::string_view seq, SinkBuffer& sink_buffer);
std::uint64_t run_var_kmer_unchecked_vectors(std::string_view seq, SinkBuffer& sink_buffer);
std::uint64_t run_var_spaced_kmer_checked(
    std::string_view seq, ScalingMaskForms const& mask_forms, SinkBuffer& sink_buffer
);
std::uint64_t run_var_spaced_kmer_checked_vectors(
    std::string_view seq, ScalingMaskForms const& mask_forms, SinkBuffer& sink_buffer
);
std::uint64_t run_var_spaced_kmer_unchecked(
    std::string_view seq, ScalingMaskForms const& mask_forms, SinkBuffer& sink_buffer
);
std::uint64_t run_var_spaced_kmer_unchecked_vectors(
    std::string_view seq, ScalingMaskForms const& mask_forms, SinkBuffer& sink_buffer
);
