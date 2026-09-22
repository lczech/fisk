#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

/**
 * @brief Benchmark different implementations to extract and iterate all k-mers in a sequence.
 *
 * The main differences between functions are how the characters are encoded into two bit encoding
 * (ifs, switch, ascii mangling, lookup table). Furthermore, we test both checked and uncheckd
 * variants (are the characters in `ACGT` - throw an exception if not), as the check adds runtime,
 * and exception handling might also cause the compiler to emit different inlinining. Lastly, we
 * benchmark full re-extraction of each k-mer (slow) vs shifting between iterations.
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

// Kernels compared above, each defined in its own translation unit (var_*.cpp).
std::uint64_t run_var_ifs_re(std::string const& seq, std::size_t k);
std::uint64_t run_var_switch_re(std::string const& seq, std::size_t k);
std::uint64_t run_var_table_re(std::string const& seq, std::size_t k);
std::uint64_t run_var_ascii_re(std::string const& seq, std::size_t k);
std::uint64_t run_var_ifs_shift(std::string const& seq, std::size_t k);
std::uint64_t run_var_switch_shift(std::string const& seq, std::size_t k);
std::uint64_t run_var_table_shift(std::string const& seq, std::size_t k);
std::uint64_t run_var_ascii_shift(std::string const& seq, std::size_t k);
std::uint64_t run_var_simd_avx2(std::string const& seq, std::size_t k);
std::uint64_t run_var_simd_scalar(std::string const& seq, std::size_t k);
