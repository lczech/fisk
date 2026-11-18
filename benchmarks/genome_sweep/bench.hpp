#pragma once

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/core/types.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "for_each_scaling/bench.hpp"
#include "seq_data.hpp"
#include "sink.hpp"

/**
 * @brief Run the algorithms of practical use on whole sequences, such as a genome, see
 * --genome-sweep in main.cpp.
 *
 * k-mers with k = 31, and spaced k-mers with `mask`, checked, ACGT, LSB, from ASCII and from packed
 * input, the latter packed once up front with WordEncoderButterfly, as in the packed suites; the
 * packing itself is timed as a case of its own. Each case also runs the estimate, for_each_kmer()
 * or for_each_spaced_kmer(), whose choice is in the case label. The algorithms are the variants of
 * the other suites, without those that are there for comparison only, and with the SIMD archs of
 * the build; those with PEXT only where it is likely fast. Each repeat is a single pass over all
 * sequences, see main.cpp for the settings.
 */
void bench_genome_sweep(
    std::vector<InputSequence> const& sequences,
    std::string const& mask,
    std::ostream& csv_os
);

using GenomeSweepPacked
    = fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>;

// The estimate on packed input, each defined in its own translation unit (var_*.cpp), as the
// estimate on ASCII input in for_each_scaling/.
std::uint64_t run_var_kmer_packed(GenomeSweepPacked const& seq, SinkBuffer& sink_buffer);
std::uint64_t run_var_kmer_packed_vectors(GenomeSweepPacked const& seq, SinkBuffer& sink_buffer);
std::uint64_t run_var_spaced_kmer_packed(
    GenomeSweepPacked const& seq, ScalingMaskForms const& mask_forms, SinkBuffer& sink_buffer
);
std::uint64_t run_var_spaced_kmer_packed_vectors(
    GenomeSweepPacked const& seq, ScalingMaskForms const& mask_forms, SinkBuffer& sink_buffer
);
