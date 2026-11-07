#include <cstdint>
#include <string_view>
#include <vector>

#include "extractor_scaling/bench.hpp"
#include "extractor_scaling/variant.hpp"

void tune_var_spaced_kmer_unchecked_vectors(
    ScalingSpacedKmerUnchecked& extractor,
    std::vector<std::uint64_t>& sink_buffer
) {
    tune_scaling_variant</*Spaced=*/true, /*Vectors=*/true>(extractor, sink_buffer);
}

std::uint64_t run_var_spaced_kmer_unchecked_vectors(
    ScalingSpacedKmerUnchecked const& extractor,
    std::string_view seq,
    std::vector<std::uint64_t>& sink_buffer
) {
    return run_scaling_variant</*Spaced=*/true, /*Vectors=*/true>(
        extractor, seq, sink_buffer
    );
}
