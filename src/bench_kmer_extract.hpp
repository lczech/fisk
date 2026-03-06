#pragma once

#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "utils.hpp"
#include "kmer_extract.hpp"
#include "seq_enc.hpp"
#include "microbench.hpp"
#include "sys_info.hpp"

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
                "char_to_nt_ifs_throw_re",
                [&](std::string const& seq){
                    return compute_kmer_hash_reextract(seq, k, char_to_nt_ifs_throw);
                }
            ),
            bench(
                "char_to_nt_ifs_nothrow_re",
                [&](std::string const& seq){
                    return compute_kmer_hash_reextract(seq, k, char_to_nt_ifs_nothrow);
                }
            ),
            bench(
                "char_to_nt_switch_throw_re",
                [&](std::string const& seq){
                    return compute_kmer_hash_reextract(seq, k, char_to_nt_switch_throw);
                }
            ),
            bench(
                "char_to_nt_switch_nothrow_re",
                [&](std::string const& seq){
                    return compute_kmer_hash_reextract(seq, k, char_to_nt_switch_nothrow);
                }
            ),
            bench(
                "char_to_nt_table_throw_re",
                [&](std::string const& seq){
                    return compute_kmer_hash_reextract(seq, k, char_to_nt_table_throw);
                }
            ),
            bench(
                "char_to_nt_table_nothrow_re",
                [&](std::string const& seq){
                    return compute_kmer_hash_reextract(seq, k, char_to_nt_table_nothrow);
                }
            ),
            bench(
                "char_to_nt_ascii_throw_re",
                [&](std::string const& seq){
                    return compute_kmer_hash_reextract(seq, k, char_to_nt_ascii_throw);
                }
            ),
            bench(
                "char_to_nt_ascii_nothrow_re",
                [&](std::string const& seq){
                    return compute_kmer_hash_reextract(seq, k, char_to_nt_ascii_nothrow);
                }
            ),

            // Shift bits
            bench(
                "char_to_nt_ifs_throw_shift",
                [&](std::string const& seq){
                    return compute_kmer_hash(seq, k, char_to_nt_ifs_throw);
                }
            ),
            bench(
                "char_to_nt_ifs_nothrow_shift",
                [&](std::string const& seq){
                    return compute_kmer_hash(seq, k, char_to_nt_ifs_nothrow);
                }
            ),
            bench(
                "char_to_nt_switch_throw_shift",
                [&](std::string const& seq){
                    return compute_kmer_hash(seq, k, char_to_nt_switch_throw);
                }
            ),
            bench(
                "char_to_nt_switch_nothrow_shift",
                [&](std::string const& seq){
                    return compute_kmer_hash(seq, k, char_to_nt_switch_nothrow);
                }
            ),
            bench(
                "char_to_nt_table_throw_shift",
                [&](std::string const& seq){
                    return compute_kmer_hash(seq, k, char_to_nt_table_throw);
                }
            ),
            bench(
                "char_to_nt_table_nothrow_shift",
                [&](std::string const& seq){
                    return compute_kmer_hash(seq, k, char_to_nt_table_nothrow);
                }
            ),
            bench(
                "char_to_nt_ascii_throw_shift",
                [&](std::string const& seq){
                    return compute_kmer_hash(seq, k, char_to_nt_ascii_throw);
                }
            ),
            bench(
                "char_to_nt_ascii_nothrow_shift",
                [&](std::string const& seq){
                    return compute_kmer_hash(seq, k, char_to_nt_ascii_nothrow);
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
