#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/seq_pack/seq_pack.hpp"
#include "sink.hpp"

using PackedAcgtMsb = fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>;
using PackedAcgtLsb = fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>;
using PackedActgMsb = fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>;
using PackedActgLsb = fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>;

/**
 * @brief Benchmark different implementations to extract and iterate all k-mers in a sequence.
 *
 * The main differences between functions are how the characters are encoded into two bit encoding
 * (ifs, switch, ascii mangling, lookup table). Furthermore, we test both checked and uncheckd
 * variants (are the characters in `ACGT` - throw an exception if not), as the check adds runtime,
 * and exception handling might also cause the compiler to emit different inlinining. Lastly, we
 * benchmark full re-extraction of each k-mer (slow) vs shifting between iterations.
 *
 * Additionally, for input known to be valid: for_each_kmer_ascii_assume_valid(), which packs the
 * sequence in chunks and extracts from those, against packing the whole sequence first and then
 * extracting via for_each_kmer_packed_aligned(). Both run for both Encodings and both Layouts, and
 * pack with the butterfly word encoder. The Encoding only affects the packing step (ACTG needs no
 * bit fold, unlike ACGT), not the extraction from the packed sequence.
 */
void bench_kmer_extract(
    std::vector<std::string> const& sequences,
    std::size_t k_min,
    std::size_t k_max,
    std::ostream& csv_os
);

// Test all valid k-mer sizes.
void bench_kmer_extract(
    std::vector<std::string> const& sequences,
    std::ostream& csv_os
);

// Kernels compared above, each defined in its own translation unit (var_*.cpp). Each one
// constructs its own local Sink (see sink.hpp) via make_sink(sink_buffer), so that Sum's
// accumulation can still auto-vectorize the way plain `hash += kmer_value(kmer)` did before -- see
// the matching comment in bit_extract_weights/bench.hpp for why a Sink built elsewhere and passed
// in by reference would defeat that.
std::uint64_t run_var_ifs_re(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_switch_re(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_table_re(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_ascii_re(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_ifs_shift(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_switch_shift(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_table_shift(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_ascii_shift(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_avx2(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_scalar(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);

// Non-validating variants. pack_then_extract takes its PackedSequence buffer from the caller, so
// that it is reused across calls rather than reallocated each time.
std::uint64_t run_var_acgt_msb_chunked_assume_valid(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_acgt_lsb_chunked_assume_valid(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_actg_msb_chunked_assume_valid(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_actg_lsb_chunked_assume_valid(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_acgt_msb_pack_then_extract(
    std::string const& seq,
    std::size_t k,
    PackedAcgtMsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_acgt_lsb_pack_then_extract(
    std::string const& seq,
    std::size_t k,
    PackedAcgtLsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_actg_msb_pack_then_extract(
    std::string const& seq,
    std::size_t k,
    PackedActgMsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_actg_lsb_pack_then_extract(
    std::string const& seq,
    std::size_t k,
    PackedActgLsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
