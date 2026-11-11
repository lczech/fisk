#include <cstdint>
#include <string_view>
#include <vector>

#include "extractor_scaling/bench.hpp"
#include "extractor_scaling/variant.hpp"

void tune_var_spaced_kmer_checked(
    ScalingSpacedKmerChecked& extractor,
    SinkBuffer& sink_buffer
) {
    tune_scaling_variant</*Spaced=*/true, /*Vectors=*/false>(extractor, sink_buffer);
}

std::uint64_t run_var_spaced_kmer_checked(
    ScalingSpacedKmerChecked const& extractor,
    std::string_view seq,
    SinkBuffer& sink_buffer
) {
    return run_scaling_variant</*Spaced=*/true, /*Vectors=*/false>(
        extractor, seq, sink_buffer
    );
}
