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
std::uint64_t run_var_acgt_ifs(std::string const& seq, std::vector<std::uint64_t>& sink_buffer);
std::uint64_t run_var_acgt_switch(std::string const& seq, std::vector<std::uint64_t>& sink_buffer);
std::uint64_t run_var_acgt_table(std::string const& seq, std::vector<std::uint64_t>& sink_buffer);
std::uint64_t run_var_acgt_ascii_validate(
    std::string const& seq,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_acgt_ascii_assume_valid(
    std::string const& seq,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_actg_ifs(std::string const& seq, std::vector<std::uint64_t>& sink_buffer);
std::uint64_t run_var_actg_switch(std::string const& seq, std::vector<std::uint64_t>& sink_buffer);
std::uint64_t run_var_actg_table(std::string const& seq, std::vector<std::uint64_t>& sink_buffer);
std::uint64_t run_var_actg_ascii_validate(
    std::string const& seq,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_actg_ascii_assume_valid(
    std::string const& seq,
    std::vector<std::uint64_t>& sink_buffer
);
