#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "sink.hpp"

/**
 * @brief Benchmark different implementations to extract and iterate all k-mers in a sequence.
 *
 * The main differences between functions are how the characters are encoded into two bit encoding
 * (ifs, switch, ascii mangling, lookup table), and full re-extraction of each k-mer (slow) vs
 * rolling between positions. All of them check the characters, and skip the k-mers with an invalid
 * one. The rolling loops with AVX2 and SWAR encode and check several characters at a time.
 */
void bench_kmer_extract_ascii_checked(
    std::vector<std::string> const& sequences,
    std::size_t k_min,
    std::size_t k_max,
    std::ostream& csv_os
);

// Test all valid k-mer sizes.
void bench_kmer_extract_ascii_checked(
    std::vector<std::string> const& sequences,
    std::ostream& csv_os
);

// Kernels compared above, each defined in its own translation unit (var_*.cpp). Each one
// constructs its own local Sink (see sink.hpp) via make_sink(sink_buffer), so that Sum's
// accumulation can still auto-vectorize the way plain `hash += kmer_value(kmer)` did before -- see
// the matching comment in bit_extract_weights/bench.hpp for why a Sink built elsewhere and passed
// in by reference would defeat that.
std::uint64_t run_var_reextract_scalar_ifs_acgt_msb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_reextract_scalar_switch_acgt_msb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_reextract_scalar_table_acgt_msb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_reextract_scalar_ascii_acgt_msb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_ifs_acgt_msb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_switch_acgt_msb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_table_acgt_msb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_ascii_acgt_msb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_avx2_acgt_msb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_swar_acgt_msb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);

// Variants emitting Layout::kLSB, to compare against their MSB counterparts.
std::uint64_t run_var_rolling_scalar_table_acgt_lsb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_ascii_acgt_lsb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_avx2_acgt_lsb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_swar_acgt_lsb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);

// Variants emitting Encoding::kACTG, to compare against their ACGT counterparts.
std::uint64_t run_var_rolling_scalar_ascii_actg_msb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_avx2_actg_msb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_swar_actg_msb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
