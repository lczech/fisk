#pragma once

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "sink.hpp"

/**
 * @brief Benchmark different implementations for encoding ASCII chars into the two bit encoding.
 *
 * This tests both variants of the implementations, those that check that the character is valid
 * in the encoding, and those that assume it is. The former will usually be more important in
 * practice on input data, while the latter might be used internally after parsing has already
 * been done. Tested for both encodings (ACGT and ACTG), one suite each.
 */
void bench_char_encoder(std::vector<std::string> const& sequences, std::ostream& csv_os);

// Kernels compared above, each defined in its own translation unit (var_*.cpp). Each one
// constructs its own local Sink (see sink.hpp) via make_sink(sink_buffer), so that Sum's
// accumulation can still auto-vectorize the way plain `h += encoder(c)` did before -- see the
// matching comment in bit_extract_weights/bench.hpp for why a Sink built elsewhere and passed in
// by reference would defeat that.
std::uint64_t run_var_ifs_acgt(std::string const& seq, std::vector<std::uint64_t>& sink_buffer);
std::uint64_t run_var_switch_acgt(std::string const& seq, std::vector<std::uint64_t>& sink_buffer);
std::uint64_t run_var_table_acgt(std::string const& seq, std::vector<std::uint64_t>& sink_buffer);
std::uint64_t run_var_ascii_checked_acgt(
    std::string const& seq,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_ascii_unchecked_acgt(
    std::string const& seq,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_ifs_actg(std::string const& seq, std::vector<std::uint64_t>& sink_buffer);
std::uint64_t run_var_switch_actg(std::string const& seq, std::vector<std::uint64_t>& sink_buffer);
std::uint64_t run_var_table_actg(std::string const& seq, std::vector<std::uint64_t>& sink_buffer);
std::uint64_t run_var_ascii_checked_actg(
    std::string const& seq,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_ascii_unchecked_actg(
    std::string const& seq,
    std::vector<std::uint64_t>& sink_buffer
);
