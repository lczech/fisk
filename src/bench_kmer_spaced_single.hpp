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
#include "kmer_spaced_simd.hpp"
#include "kmer_spaced_selector.hpp"
#include "seq_enc.hpp"
#include "microbench.hpp"
#include "sys_info.hpp"

/**
 * @brief Benchmark spaced k-mer extract with single masks.
 */
inline void bench_kmer_spaced_single(
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
    std::string const suite_title = "kmer_spaced_single";
    std::cout << "\n=== spaced k-mer extract single mask ===\n";
    std::cout << "rounds=" << rounds << ", repeats=" << repeats << "\n";

    // Prepare csv output file with benchmark results
    write_csv_header(csv_os);

    // Run a benchmark for each mask
    for( size_t m = 0; m < masks.size(); ++m) {
        auto const k = masks[m].size();
        if( stdout_is_terminal() ) {
            std::cout << "\rmask ";
            std::cout << std::setw(2) << (m+1) << " / " << masks.size() << "\n";
        }

        // Prepare masks for all implementations as needed
        auto const naive_mask = prepare_naive_mask(masks[m]);
        auto const bit_ext_mask = prepare_spaced_kmer_bit_extract_mask(masks[m]);
        auto const bit_ext_block_mask = bit_extract_block_table_preprocess(bit_ext_mask);
        auto const bit_ext_butterfly_table = bit_extract_butterfly_table_preprocess(bit_ext_mask);

        // Just to test the selector
        std::cout << "fastest mode: ";
        std::cout << spaced_kmer_mode_name( spaced_kmer_selector( bit_ext_mask, k )) << "\n";

        // simd kernels
        BitExtractKernelButterflyScalar simd_bf_scalar_kernel(bit_ext_mask);
        BitExtractKernelBlockScalar     simd_bt_scalar_kernel(bit_ext_mask);
        #if defined(FISK_HAS_SSE2)
        BitExtractKernelButterflySSE2   simd_bf_sse2_kernel(bit_ext_mask);
        BitExtractKernelBlockSSE2       simd_bt_sse2_kernel(bit_ext_mask);
        #endif
        #if defined(FISK_HAS_AVX2)
        BitExtractKernelButterflyAVX2   simd_bf_avx2_kernel(bit_ext_mask);
        BitExtractKernelBlockAVX2       simd_bt_avx2_kernel(bit_ext_mask);
        #endif
        #if defined(FISK_HAS_AVX512)
        BitExtractKernelButterflyAVX512 simd_bf_avx512_kernel(bit_ext_mask);
        BitExtractKernelBlockAVX512     simd_bt_avx512_kernel(bit_ext_mask);
        #endif
        #if defined(FISK_HAS_NEON)
        BitExtractKernelButterflyNEON   simd_bf_neon_kernel(bit_ext_mask);
        BitExtractKernelBlockNEON       simd_bt_neon_kernel(bit_ext_mask);
        #endif
        #if defined(FISK_HAS_BMI2)
        BitExtractKernelPEXT            simd_pext_kernel(bit_ext_mask);
        #endif

        // Prepare a benchmark with repititions
        Microbench<std::string> suite(suite_title);
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

            // naive, as baseline and validity check
            bench(
                "missh",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_naive(
                        seq, k, naive_mask, compute_spaced_kmer_missh
                    );
                }
            ),
            bench(
                "naive",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_naive(
                        seq, k, naive_mask, compute_spaced_kmer_naive
                    );
                }
            ),

            // char_to_nt_table
            #if defined(FISK_HAS_BMI2)
            bench(
                "pext",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash(
                        seq, k, BitExtractMask(bit_ext_mask), char_to_nt_table, bit_extract_pext
                    );
                }
            ),
            #endif
            bench(
                "bitloop",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash(
                        seq, k, BitExtractMask(bit_ext_mask), char_to_nt_table, bit_extract_bitloop
                    );
                }
            ),
            bench(
                "byte_table",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash(
                        seq, k, BitExtractMask(bit_ext_mask), char_to_nt_table, bit_extract_byte_table
                    );
                }
            ),
            bench(
                "block_table",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash(
                        seq, k, bit_ext_block_mask, char_to_nt_table, bit_extract_block_table
                    );
                }
            ),
            bench(
                "block_table_unrolled2",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash(
                        seq, k, bit_ext_block_mask, char_to_nt_table, bit_extract_block_table_unrolled<2>
                    );
                }
            ),
            bench(
                "block_table_unrolled4",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash(
                        seq, k, bit_ext_block_mask, char_to_nt_table, bit_extract_block_table_unrolled<4>
                    );
                }
            ),
            bench(
                "block_table_unrolled8",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash(
                        seq, k, bit_ext_block_mask, char_to_nt_table, bit_extract_block_table_unrolled<8>
                    );
                }
            ),
            bench(
                "butterfly_table",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash(
                        seq, k, bit_ext_butterfly_table, char_to_nt_table, bit_extract_butterfly_table
                    );
                }
            ),

            // simd kernels
            #if defined(FISK_HAS_SSE2)
            bench(
                "simd_butterfly_table_sse2",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_simd(
                        seq, k, simd_bf_sse2_kernel
                    );
                }
            ),
            bench(
                "simd_block_table_sse2",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_simd(
                        seq, k, simd_bt_sse2_kernel
                    );
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX2)
            bench(
                "simd_butterfly_table_avx2",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_simd(
                        seq, k, simd_bf_avx2_kernel
                    );
                }
            ),
            bench(
                "simd_block_table_avx2",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_simd(
                        seq, k, simd_bt_avx2_kernel
                    );
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX512)
            bench(
                "simd_butterfly_table_avx512",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_simd(
                        seq, k, simd_bf_avx512_kernel
                    );
                }
            ),
            bench(
                "simd_block_table_avx512",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_simd(
                        seq, k, simd_bt_avx512_kernel
                    );
                }
            ),
            #endif
            #if defined(FISK_HAS_NEON)
            bench(
                "simd_butterfly_table_neon",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_simd(
                        seq, k, simd_bf_neon_kernel
                    );
                }
            ),
            bench(
                "simd_block_table_neon",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_simd(
                        seq, k, simd_bt_neon_kernel
                    );
                }
            ),
            #endif
            #if defined(FISK_HAS_BMI2)
            bench(
                "simd_pext",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_simd(
                        seq, k, simd_pext_kernel
                    );
                }
            ),
            #endif
            bench(
                "simd_butterfly_table_scalar",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_simd(
                        seq, k, simd_bf_scalar_kernel
                    );
                }
            ),
            bench(
                "simd_block_table_scalar",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_simd(
                        seq, k, simd_bt_scalar_kernel
                    );
                }
            )

            // char_to_nt_switch
            // #if defined(FISK_HAS_BMI2)
            // bench(
            //     "pext_char_to_nt_switch",
            //     [&](std::string const& seq){
            //         return compute_spaced_kmer_hash(
            //             seq, k, bit_ext_mask, char_to_nt_switch, bit_extract_pext
            //         );
            //     }
            // ),
            // #endif
            // bench(
            //     "bitloop_char_to_nt_switch",
            //     [&](std::string const& seq){
            //         return compute_spaced_kmer_hash(
            //             seq, k, bit_ext_mask, char_to_nt_switch, bit_extract_bitloop
            //         );
            //     }
            // ),
            // bench(
            //     "byte_table_char_to_nt_switch",
            //     [&](std::string const& seq){
            //         return compute_spaced_kmer_hash(
            //             seq, k, bit_ext_mask, char_to_nt_switch, bit_extract_byte_table
            //         );
            //     }
            // ),
            // bench(
            //     "block_table_char_to_nt_switch",
            //     [&](std::string const& seq){
            //         return compute_spaced_kmer_hash(
            //             seq, k, bit_ext_block_mask, char_to_nt_switch, bit_extract_block_table
            //         );
            //     }
            // ),

            // char_to_nt_ascii
            // #if defined(FISK_HAS_BMI2)
            // bench(
            //     "pext_char_to_nt_ascii",
            //     [&](std::string const& seq){
            //         return compute_spaced_kmer_hash(
            //             seq, k, bit_ext_mask, char_to_nt_ascii, bit_extract_pext
            //         );
            //     }
            // ),
            // #endif
            // bench(
            //     "bitloop_char_to_nt_ascii",
            //     [&](std::string const& seq){
            //         return compute_spaced_kmer_hash(
            //             seq, k, bit_ext_mask, char_to_nt_ascii, bit_extract_bitloop
            //         );
            //     }
            // ),
            // bench(
            //     "byte_table_char_to_nt_ascii",
            //     [&](std::string const& seq){
            //         return compute_spaced_kmer_hash(
            //             seq, k, bit_ext_mask, char_to_nt_ascii, bit_extract_byte_table
            //         );
            //     }
            // ),
            // bench(
            //     "block_table_char_to_nt_ascii",
            //     [&](std::string const& seq){
            //         return compute_spaced_kmer_hash(
            //             seq, k, bit_ext_block_mask, char_to_nt_ascii, bit_extract_block_table
            //         );
            //     }
            // )

        );

        std::string case_label = "mask=" + std::to_string(m);
        write_csv_rows(csv_os, suite_title, case_label, results);
    }
    if( stdout_is_terminal() ) {
        std::cout << "\n";
    }
}
