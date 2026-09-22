#pragma once

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

/**
 * @brief Benchmark different implementations for encoding ASCII chars into the two bit encoding.
 *
 * This tests both variants of the implementations, those that check that the character is valid
 * in the encoding, and those that assume it is. The former will usually be more important in
 * practice on input data, while the latter might be used internally after parsing has already
 * been done. Tested for both encodings (ACGT and ACTG), one suite each.
 *
 * The barrier variants (forcing each character's code through do_not_optimize() to isolate its
 * scalar cost from auto-vectorization) are dropped here for now, pending the planned generic
 * sink redesign that will reintroduce that capability more broadly.
 */
void bench_char_encoder(std::vector<std::string> const& sequences, std::ostream& csv_os);

// Kernels compared above, each defined in its own translation unit (var_*.cpp).
std::uint64_t run_var_acgt_ifs(std::string const& seq);
std::uint64_t run_var_acgt_switch(std::string const& seq);
std::uint64_t run_var_acgt_table(std::string const& seq);
std::uint64_t run_var_acgt_ascii_validate(std::string const& seq);
std::uint64_t run_var_acgt_ascii_assume_valid(std::string const& seq);
std::uint64_t run_var_actg_ifs(std::string const& seq);
std::uint64_t run_var_actg_switch(std::string const& seq);
std::uint64_t run_var_actg_table(std::string const& seq);
std::uint64_t run_var_actg_ascii_validate(std::string const& seq);
std::uint64_t run_var_actg_ascii_assume_valid(std::string const& seq);
