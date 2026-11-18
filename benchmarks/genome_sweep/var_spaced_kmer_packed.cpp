#include <cstdint>

#include "genome_sweep/bench.hpp"
#include "genome_sweep/variant.hpp"

std::uint64_t run_var_spaced_kmer_packed(
    GenomeSweepPacked const& seq, ScalingMaskForms const& mask_forms, SinkBuffer& sink_buffer
) {
    return run_genome_sweep_spaced_kmer_packed</*Vectors=*/false>(seq, mask_forms, sink_buffer);
}
