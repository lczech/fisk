#pragma once

#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "fs_utils.hpp"
#include "kmer_extract.hpp"
#include "seq_enc.hpp"
#include "microbench.hpp"
#include "sys_info.hpp"

template<typename Enc>
inline std::uint64_t for_each_kmer_2bit_xor(std::string_view seq, std::size_t k, Enc&& enc)
{
    // Simple wrapper around the main loop function whcih also keeps track of a "hash"
    // by xor-ing all k-mers, just as a validity check that all implementations give the same.
    std::uint64_t hash = 0;

    for_each_kmer_2bit(
        std::string_view(seq),
        k,
        enc,
        [&](std::uint64_t kmer_word) {
            // Simple order-independent checksum.
            // All implementations must use the same aggregation so sinks match.
            hash ^= kmer_word;
        }
    );

    return hash;
}

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

    std::size_t const rounds = 1;
    std::size_t const repeats = 8;

    // User output
    std::cout << "\n=== k-mer extract ===\n";
    std::cout << "rounds=" << rounds << ", repeats=" << repeats << "\n";

    // Prepare csv output file with benchmark results
    write_csv_header(csv_os);

    // Run a benchmark for each valid k.
    for( size_t k = k_min; k <= k_max; ++k) {
        if( stdout_is_terminal() ) {
            std::cout << "\rk "
                << std::setw(2) << k << " / " << (k_max - k_min)
                << std::flush;
        }

        Microbench<std::string> suite("kmer_extract");
        suite
            .rounds(rounds)
            .repeats(repeats)
            .units_fn([k](std::string const& seq) {
                // 1 unit per k-mer
                return static_cast<double>(seq.size() - k + 1);
            });

        auto results = suite.run(
            sequences, // vector<std::string>
            bench(
                "char_to_nt_switch",
                [&](std::string const& seq){
                    return for_each_kmer_2bit_xor(seq, k, char_to_nt_switch);
                }
            ),
            bench(
                "char_to_nt_table",
                [&](std::string const& seq){
                    return for_each_kmer_2bit_xor(seq, k, char_to_nt_table);
                }
            ),
            bench(
                "char_to_nt_ascii",
                [&](std::string const& seq){
                    return for_each_kmer_2bit_xor(seq, k, char_to_nt_ascii);
                }
            )
        );

        std::string case_label = "k=" + std::to_string(k);
        write_csv_rows(csv_os, "kmer_extract", case_label, results);
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
