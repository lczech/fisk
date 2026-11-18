#include <cstdint>

#include "genome_sweep/bench.hpp"
#include "genome_sweep/variant.hpp"

std::uint64_t run_var_kmer_packed(GenomeSweepPacked const& seq, SinkBuffer& sink_buffer)
{
    return run_genome_sweep_kmer_packed</*Vectors=*/false>(seq, sink_buffer);
}
