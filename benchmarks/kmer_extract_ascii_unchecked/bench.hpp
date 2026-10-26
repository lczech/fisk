#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "sink.hpp"

using PackedAcgtMsb = fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>;
using PackedAcgtLsb = fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>;
using PackedActgMsb = fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>;
using PackedActgLsb = fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>;

/**
 * @brief Benchmark the extraction of all k-mers of a sequence from ASCII input without checks.
 *
 * For input known to be valid: for_each_kmer_ascii_unchecked_chunked_scalar(), which
 * packs the sequence in chunks and extracts from those, against packing the whole sequence first
 * and then extracting via for_each_kmer_packed_unchecked_bytewise_scalar(). Both run for both
 * Encodings and both Layouts, and pack with the butterfly word encoder. The Encoding only affects
 * the packing step (ACTG needs no bit fold, unlike ACGT), not the extraction from the packed
 * sequence.
 */
void bench_kmer_extract_ascii_unchecked(
    std::vector<std::string> const& sequences,
    std::size_t k_min,
    std::size_t k_max,
    std::ostream& csv_os
);

// Test all valid k-mer sizes.
void bench_kmer_extract_ascii_unchecked(
    std::vector<std::string> const& sequences,
    std::ostream& csv_os
);

// Kernels compared above, each defined in its own translation unit (var_*.cpp). Each one
// constructs its own local Sink (see sink.hpp) via make_sink(sink_buffer), so that Sum's
// accumulation can still auto-vectorize the way plain `hash += kmer_value(kmer)` did before -- see
// the matching comment in bit_extract_weights/bench.hpp for why a Sink built elsewhere and passed
// in by reference would defeat that.
// The scratch-buffer variants take caller-owned PackedSequence storage,
// so it is reused across calls rather than reallocated each time.
std::uint64_t run_var_chunked_scalar_threadlocal_acgt_msb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_scalar_threadlocal_acgt_lsb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_scalar_threadlocal_actg_msb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_scalar_threadlocal_actg_lsb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_scalar_scratch_acgt_msb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtMsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_scalar_scratch_acgt_lsb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtLsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_scalar_scratch_actg_msb(
    std::string const& seq,
    std::size_t k,
    PackedActgMsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_scalar_scratch_actg_lsb(
    std::string const& seq,
    std::size_t k,
    PackedActgLsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_pack_then_bytewise_scalar_acgt_msb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtMsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_pack_then_bytewise_scalar_acgt_lsb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtLsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_pack_then_bytewise_scalar_actg_msb(
    std::string const& seq,
    std::size_t k,
    PackedActgMsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_pack_then_bytewise_scalar_actg_lsb(
    std::string const& seq,
    std::size_t k,
    PackedActgLsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
