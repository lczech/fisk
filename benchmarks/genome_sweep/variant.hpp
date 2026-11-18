#pragma once

#include <cstdint>

#include "fisk/core/types.hpp"
#include "fisk/extraction/for_each.hpp"
#include "for_each_scaling/variant.hpp"
#include "genome_sweep/bench.hpp"
#include "sink.hpp"

// =================================================================================================
//     Variants of the Genome Sweep
// =================================================================================================

// The estimate on packed input, for the var_*.cpp, with the callbacks of the scaling suite, see
// make_scaling_callback() (for_each_scaling/variant.hpp).

template <bool Vectors>
std::uint64_t run_genome_sweep_kmer_packed(GenomeSweepPacked const& seq, SinkBuffer& sink_buffer)
{
    auto sink = make_sink(sink_buffer);
    fisk::for_each_kmer(seq, kScalingK, make_scaling_callback</*Spaced=*/false, Vectors>(sink));
    return sink.finalize();
}

template <bool Vectors>
std::uint64_t run_genome_sweep_spaced_kmer_packed(
    GenomeSweepPacked const& seq,
    ScalingMaskForms const& mask_forms,
    SinkBuffer& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    fisk::for_each_spaced_kmer(
        seq, mask_forms, make_scaling_callback</*Spaced=*/true, Vectors>(sink)
    );
    return sink.finalize();
}
