#pragma once

#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "utils.hpp"
#include "fisk/kmer_extract/kmer_extract.hpp"
#include "fisk/kmer_extract/simd.hpp"
#include "fisk/core/seq_enc.hpp"
#include "microbench.hpp"

// =================================================================================================
//     Sum Hashing
// =================================================================================================

// Benchmark sinks for the for_each_kmer*() functions (kmer_extract/kmer_extract.hpp and
// kmer_extract/simd.hpp): sum every emitted k-mer into `hash`, so that the compiler cannot
// optimize the extraction away as dead code, and so that all benchmarked variants below can be
// checked against the same aggregate. Not library functionality, hence living here rather than in
// those headers; see bench_kmer_extract_packed.hpp's compute_kmer_hash_packed_*() for the same
// pattern applied to the packed extractors.

template<typename Enc>
inline std::uint64_t compute_kmer_hash(std::string_view seq, std::size_t k, Enc&& enc)
{
    std::uint64_t hash = 0;
    for_each_kmer_rolling(seq, k, enc, [&](std::uint64_t kmer_word) { hash += kmer_word; });
    return hash;
}

template<typename Enc>
inline std::uint64_t compute_kmer_hash_reextract(std::string_view seq, std::size_t k, Enc&& enc)
{
    std::uint64_t hash = 0;
    for_each_kmer_reextract(seq, k, enc, [&](std::uint64_t kmer_word) { hash += kmer_word; });
    return hash;
}

inline std::uint64_t compute_kmer_hash_simd(std::string_view seq, std::size_t k)
{
    std::uint64_t hash = 0;
    for_each_kmer_simd(seq, k, [&](std::uint64_t kmer_word) { hash += kmer_word; });
    return hash;
}

inline std::uint64_t compute_kmer_hash_simd_scalar(std::string_view seq, std::size_t k)
{
    std::uint64_t hash = 0;
    for_each_kmer_simd_scalar(seq, k, [&](std::uint64_t kmer_word) { hash += kmer_word; });
    return hash;
}

/**
 * @brief Benchmark different implementations to extract and iterate all k-mers in a sequence.
 *
 * The main differences between functions are how the characters are encoded into two bit encoding
 * (ifs, switch, ascii mangling, lookup table). Furthermore, we test both checked and uncheckd
 * variants (are the characters in `ACGT` - throw an exception if not), as the check adds runtime,
 * and exception handling might also cause the compiler to emit different inlinining. Lastly, we
 * benchmark full re-extraction of each k-mer (slow) vs shifting between iterations.
 */
inline void bench_kmer_extract(
    std::vector<std::string> const& sequences,
    size_t k_min,
    size_t k_max,
    std::ostream& csv_os
) {
    // Boundary checks
    if( k_min < 1 || k_min > 32 || k_max < 1 || k_max > 32 ) {
        throw std::runtime_error( "Invalid k outside of [1, 32]" );
    }
    if( k_min > k_max ) {
        throw std::runtime_error( "Invalid k_min > k_max" );
    }

    std::size_t const rounds = 8;
    std::size_t const repeats = 8;

    // User output
    std::string const suite_title = "kmer_extract";
    std::cout << "\n=== k-mer extract ===\n";
    std::cout << "rounds=" << rounds << ", repeats=" << repeats << "\n";

    // Prepare csv output file with benchmark results
    write_csv_header(csv_os);

    // Run a benchmark for each valid k.
    for( size_t k = k_min; k <= k_max; ++k) {
        if( stdout_is_terminal() ) {
            std::cout << "\rk " << std::setw(2) << k << std::flush;
        }

        Microbench<std::string> suite(suite_title);
        suite
            .rounds(rounds)
            .repeats(repeats)
            .units_fn([k](std::string const& seq) {
                // 1 unit per k-mer
                return static_cast<double>(seq.size() - k + 1);
            });

        auto results = suite.run(
            sequences, // vector<std::string>

            // Full re-extract
            bench(
                "char_to_nt_ifs_re",
                [&](std::string const& seq){
                    return compute_kmer_hash_reextract(seq, k, char_to_nt_ifs_acgt);
                }
            ),
            bench(
                "char_to_nt_switch_re",
                [&](std::string const& seq){
                    return compute_kmer_hash_reextract(seq, k, char_to_nt_switch_acgt);
                }
            ),
            bench(
                "char_to_nt_table_re",
                [&](std::string const& seq){
                    return compute_kmer_hash_reextract(seq, k, char_to_nt_table_acgt);
                }
            ),
            bench(
                "char_to_nt_ascii_re",
                [&](std::string const& seq){
                    return compute_kmer_hash_reextract(seq, k, char_to_nt_ascii_acgt);
                }
            ),

            // Shift bits
            bench(
                "char_to_nt_ifs_shift",
                [&](std::string const& seq){
                    return compute_kmer_hash(seq, k, char_to_nt_ifs_acgt);
                }
            ),
            bench(
                "char_to_nt_switch_shift",
                [&](std::string const& seq){
                    return compute_kmer_hash(seq, k, char_to_nt_switch_acgt);
                }
            ),
            bench(
                "char_to_nt_table_shift",
                [&](std::string const& seq){
                    return compute_kmer_hash(seq, k, char_to_nt_table_acgt);
                }
            ),
            bench(
                "char_to_nt_ascii_shift",
                [&](std::string const& seq){
                    return compute_kmer_hash(seq, k, char_to_nt_ascii_acgt);
                }
            ),

            // SIMD, currently only AVX2 and scalar, for testing
            bench(
                "char_to_nt_simd_avx2",
                [&](std::string const& seq){
                    return compute_kmer_hash_simd(seq, k);
                }
            ),
            bench(
                "char_to_nt_simd_scalar",
                [&](std::string const& seq){
                    return compute_kmer_hash_simd_scalar(seq, k);
                }
            )
        );

        std::string case_label = "k=" + std::to_string(k);
        write_csv_rows(csv_os, suite_title, case_label, results);
    }
    if( stdout_is_terminal() ) {
        std::cout << "\n";
    }
}

inline void bench_kmer_extract(
    std::vector<std::string> const& sequences,
    std::ostream& csv_os
) {
    // Test all valid k-mer sizes
    bench_kmer_extract( sequences, 1, 32, csv_os );
}
