#pragma once

#include <cstdint>
#include <iostream>
#include <functional>
#include <fstream>
#include <string>
#include <vector>
#include <utility>

#include "utils.hpp"
#include "kmer_clark.hpp"
#include "kmer_spaced.hpp"
#include "seq_enc.hpp"
#include "microbench.hpp"
#include "sys_info.hpp"

inline void bench_kmer_clark(
    std::vector<std::string> const& sequences,
    std::ostream& csv_os
) {
    std::size_t const rounds = 4;
    std::size_t const repeats = 16;
    size_t const k = 31;

    // User output
    std::cout << "\n=== spaced k-mer clark ===\n";
    std::cout << "rounds=" << rounds << ", repeats=" << repeats << "\n";

    // Prepare csv output file with benchmark results
    write_csv_header(csv_os);

    // We store the masks here, in a simplified form with just the names.
    // The actual masks are hard-coded in clark.
    std::vector<std::string> masks = {{ "T295", "T38570", "T58570" }};

    // Prepare a benchmark with repititions
    Microbench<std::string> suite("kmer_clark");
    suite
        .rounds(rounds)
        .repeats(repeats)
        .units_fn([](std::string const& seq) {
            // 1 unit per original (unspaced) k-mer
            return static_cast<double>(seq.size() - k + 1);
        });

    // Run the benchmark for all algorithms
    auto results = suite.run(
        sequences, // vector<std::string>

        // clark
        bench(
            "clark",
            [&](std::string const& seq){
                return clark_getObjectsDataComputeFull( seq, masks );
            }
        ),
        bench(
            "clark_improved",
            [&](std::string const& seq){
                return clark_improved( seq );
            }
        )
    );

    std::string case_label = "n/a";
    write_csv_rows(csv_os, "kmer_clark", case_label, results);
}

/*
inline void bench_kmer_spaced_clark(
    std::vector<std::string> const& sequences,
    std::ostream& csv_os
) {
    std::size_t const rounds = 1;
    std::size_t const repeats = 8;

    // User output
    std::cout << "\n=== spaced k-mer clark ===\n";
    std::cout << "rounds=" << rounds << ", repeats=" << repeats << "\n";

    // Prepare csv output file with benchmark results
    write_csv_header(csv_os);

    // The base clark implementation expects the seed name as a string.
    // Very inefficient, but let's test this as well, for completeness.
    // To this end, we create a set of pairs, with the name and mask.
    std::vector<std::pair<std::string, std::string>> clark_masks = {{
        {"T295",   "1111011101110010111001011011111"},
        {"T38570", "1111101011100101101110011011111"},
        {"T58570", "1111101001110101101100111011111"},
    }};

    // Run a benchmark for each mask
    for( size_t m = 0; m < clark_masks.size(); ++m) {
        if( stdout_is_terminal() ) {
            std::cout << "\rmask "
                << std::setw(2) << (m+1) << " / " << clark_masks.size()
                << std::flush;
        }
        auto const k = clark_masks[m].second.size();

        // Prepare masks for all implementations as needed
        auto const comin_mask = comin_prepare_mask(clark_masks[m].second);
        auto const pext_mask = pext_prepare_kmer_mask(clark_masks[m].second);
        auto const pext_block_mask = pext_sw_block_table_preprocess_u64(pext_mask);

        // Set up a fast pointer to the correct clark function,
        // so that we do not need to switch in every iteration.
        // std::function<void(size_t, uint64_t const&, uint64_t&)> clark_fct;
        using ClarkFunc = void(*)(size_t, uint64_t const&, uint64_t&);
        ClarkFunc clark_fct = nullptr;
        switch( m ) {
            case 0: {
                clark_fct = [](size_t, uint64_t const& kmer, uint64_t& result){
                    clark_getSpacedSeedOPTSS95s2_improved(kmer, result);
                };
                break;
            }
            case 1: {
                clark_fct = [](size_t, uint64_t const& kmer, uint64_t& result){
                    clark_getSpacedSeedT38570_improved(kmer, result);
                };
                break;
            }
            case 2: {
                clark_fct = [](size_t, uint64_t const& kmer, uint64_t& result){
                    clark_getSpacedSeedT58570_improved(kmer, result);
                };
                break;
            }
            default: {
                throw std::invalid_argument( "Invalid clark mask iteration.");
            }
        }

        // To get even faster results, remove the pointer completely,
        // so that the compiler can inline the call.
        auto clark_inlined = [&](std::string const& seq){
            switch( m ) {
                case 0: {
                    return clark_compute_sequence_hash(
                        k, m, seq, [](size_t, uint64_t const& kmer, uint64_t& result)
                        {
                            clark_getSpacedSeedOPTSS95s2_improved(kmer, result);
                        }
                    );
                    break;
                }
                case 1: {
                    return clark_compute_sequence_hash(
                        k, m, seq, [](size_t, uint64_t const& kmer, uint64_t& result)
                        {
                            clark_getSpacedSeedT38570_improved(kmer, result);
                        }
                    );
                    break;
                }
                case 2: {
                    return clark_compute_sequence_hash(
                        k, m, seq, [](size_t, uint64_t const& kmer, uint64_t& result)
                        {
                            clark_getSpacedSeedT58570_improved(kmer, result);
                        }
                    );
                    break;
                }
                default: {
                    throw std::invalid_argument( "Invalid clark mask iteration.");
                }
            }
        };

        // Prepare a benchmark with repititions
        Microbench<std::string> suite("kmer_spaced_clark");
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

            // clark
            bench(
                "clark",
                [&](std::string const& seq){
                    return clark_compute_sequence_hash(
                        k, clark_masks[m].first, seq, clark_getSpacedSeed
                    );
                }
            ),
            bench(
                "clark_improved",
                [&](std::string const& seq){
                    return clark_compute_sequence_hash(
                        k, m, seq, clark_fct
                    );
                }
            ),
            bench(
                "clark_inlined",
                clark_inlined
            ),

            // char_to_nt_table
            #if defined(HAVE_BMI2)
            bench(
                "pext_hw_bmi2",
                [&](std::string const& seq){
                    return pext_compute_sequence_hash(
                        k, pext_mask, seq, char_to_nt_table, pext_hw_bmi2_u64
                    );
                }
            ),
            #endif
            bench(
                "pext_sw_bitloop",
                [&](std::string const& seq){
                    return pext_compute_sequence_hash(
                        k, pext_mask, seq, char_to_nt_table, pext_sw_bitloop_u64
                    );
                }
            ),
            bench(
                "pext_sw_table8",
                [&](std::string const& seq){
                    return pext_compute_sequence_hash(
                        k, pext_mask, seq, char_to_nt_table, pext_sw_table8_u64
                    );
                }
            ),
            bench(
                "pext_sw_block_table",
                [&](std::string const& seq){
                    return pext_compute_sequence_hash(
                        k, pext_block_mask, seq, char_to_nt_table, pext_sw_block_table_u64
                    );
                }
            )

        );

        std::string case_label = "mask=" + clark_masks[m].first;
        write_csv_rows(csv_os, "kmer_spaced_clark", case_label, results);
    }
    if( stdout_is_terminal() ) {
        std::cout << "\n";
    }
}
*/
