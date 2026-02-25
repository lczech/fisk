#pragma once

#include <cstdint>
#include <iostream>
#include <functional>
#include <fstream>
#include <string>
#include <vector>
#include <utility>

#include "utils.hpp"
#include "kmer_spaced.hpp"
#include "seq_enc.hpp"
#include "microbench.hpp"
#include "sys_info.hpp"

inline void bench_kmer_spaced(
    std::vector<std::string> const& sequences,
    std::vector<std::string> const& masks,
    std::ostream& csv_os
) {
    // Boundary checks. No need to check this - we can deal with mixed mask sizes,
    // as we run them independently anyway.
    // for( size_t i = 1; i < masks.size(); ++i ) {
    //     if( masks[i].size() != masks[0].size() ) {
    //         throw std::invalid_argument( "Invalid masks of different lengths" );
    //     }
    // }

    std::size_t const rounds = 1;
    std::size_t const repeats = 8;

    // User output
    std::cout << "\n=== spaced k-mer extract ===\n";
    std::cout << "rounds=" << rounds << ", repeats=" << repeats << "\n";

    // Prepare csv output file with benchmark results
    write_csv_header(csv_os);

    // Run a benchmark for each mask
    for( size_t m = 0; m < masks.size(); ++m) {
        if( stdout_is_terminal() ) {
            std::cout << "\rmask "
                << std::setw(2) << (m+1) << " / " << masks.size()
                << std::flush;
        }
        auto const k = masks[m].size();

        // Prepare masks for all implementations as needed
        auto const comin_mask = comin_prepare_mask(masks[m]);
        auto const pext_mask = pext_prepare_kmer_mask(masks[m]);
        auto const pext_block_mask = pext_sw_block_table_preprocess_u64(pext_mask);

        // Prepare a benchmark with repititions
        Microbench<std::string> suite("kmer_spaced");
        suite
            .rounds(rounds)
            .repeats(repeats)
            .units_fn([k](std::string const& seq) {
                // 1 unit per original (unspaced) k-mer
                return static_cast<double>(seq.size() - k + 1);
            });

        // Run the benchmark for all algorithms
        auto results = suite.run(
            sequences, // vector<std::string>
            // comin
            bench(
                "comin",
                [&](std::string const& seq){
                    return comin_compute_sequence_hash(
                        k, comin_mask, seq, comin_compute_spaced_kmer
                    );
                }
            ),
            bench(
                "comin_improved",
                [&](std::string const& seq){
                    return comin_compute_sequence_hash(
                        k, comin_mask, seq, comin_compute_spaced_kmer_improved
                    );
                }
            ),

            // char_to_nt_switch
            // #if defined(HAVE_BMI2)
            // bench(
            //     "pext_hw_bmi2_u64_char_to_nt_switch",
            //     [&](std::string const& seq){
            //         return pext_compute_sequence_hash(
            //             k, pext_mask, seq, char_to_nt_switch, pext_hw_bmi2_u64
            //         );
            //     }
            // ),
            // #endif
            // bench(
            //     "pext_sw_bitloop_u64_char_to_nt_switch",
            //     [&](std::string const& seq){
            //         return pext_compute_sequence_hash(
            //             k, pext_mask, seq, char_to_nt_switch, pext_sw_bitloop_u64
            //         );
            //     }
            // ),
            // bench(
            //     "pext_sw_table8_u64_char_to_nt_switch",
            //     [&](std::string const& seq){
            //         return pext_compute_sequence_hash(
            //             k, pext_mask, seq, char_to_nt_switch, pext_sw_table8_u64
            //         );
            //     }
            // ),
            // bench(
            //     "pext_sw_block_table_u64_char_to_nt_switch",
            //     [&](std::string const& seq){
            //         return pext_compute_sequence_hash(
            //             k, pext_block_mask, seq, char_to_nt_switch, pext_sw_block_table_u64
            //         );
            //     }
            // ),

            // char_to_nt_table
            #if defined(HAVE_BMI2)
            bench(
                "pext_hw_bmi2_u64_char_to_nt_table",
                [&](std::string const& seq){
                    return pext_compute_sequence_hash(
                        k, pext_mask, seq, char_to_nt_table_throw, pext_hw_bmi2_u64
                    );
                }
            ),
            #endif
            bench(
                "pext_sw_bitloop_u64_char_to_nt_table",
                [&](std::string const& seq){
                    return pext_compute_sequence_hash(
                        k, pext_mask, seq, char_to_nt_table_throw, pext_sw_bitloop_u64
                    );
                }
            ),
            bench(
                "pext_sw_table8_u64_char_to_nt_table",
                [&](std::string const& seq){
                    return pext_compute_sequence_hash(
                        k, pext_mask, seq, char_to_nt_table_throw, pext_sw_table8_u64
                    );
                }
            ),
            bench(
                "pext_sw_block_table_u64_char_to_nt_table",
                [&](std::string const& seq){
                    return pext_compute_sequence_hash(
                        k, pext_block_mask, seq, char_to_nt_table_throw, pext_sw_block_table_u64
                    );
                }
            )

            // char_to_nt_ascii
            // #if defined(HAVE_BMI2)
            // bench(
            //     "pext_hw_bmi2_u64_char_to_nt_ascii",
            //     [&](std::string const& seq){
            //         return pext_compute_sequence_hash(
            //             k, pext_mask, seq, char_to_nt_ascii, pext_hw_bmi2_u64
            //         );
            //     }
            // ),
            // #endif
            // bench(
            //     "pext_sw_bitloop_u64_char_to_nt_ascii",
            //     [&](std::string const& seq){
            //         return pext_compute_sequence_hash(
            //             k, pext_mask, seq, char_to_nt_ascii, pext_sw_bitloop_u64
            //         );
            //     }
            // ),
            // bench(
            //     "pext_sw_table8_u64_char_to_nt_ascii",
            //     [&](std::string const& seq){
            //         return pext_compute_sequence_hash(
            //             k, pext_mask, seq, char_to_nt_ascii, pext_sw_table8_u64
            //         );
            //     }
            // ),
            // bench(
            //     "pext_sw_block_table_u64_char_to_nt_ascii",
            //     [&](std::string const& seq){
            //         return pext_compute_sequence_hash(
            //             k, pext_block_mask, seq, char_to_nt_ascii, pext_sw_block_table_u64
            //         );
            //     }
            // )

        );

        std::string case_label = "mask=" + std::to_string(m);
        write_csv_rows(csv_os, "kmer_spaced", case_label, results);
    }
    if( stdout_is_terminal() ) {
        std::cout << "\n";
    }
}
