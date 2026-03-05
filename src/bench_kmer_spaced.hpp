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
#include "seq_enc.hpp"
#include "microbench.hpp"
#include "sys_info.hpp"

/**
 * @brief Benchmark spaced k-mer extract with single masks.
 */
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
        auto const bit_ext_mask = prepare_spaced_kmer_bit_extract_mask(masks[m]);
        auto const bit_ext_block_mask = bit_extract_block_table_preprocess(bit_ext_mask);
        auto const bit_ext_network_table = bit_extract_network_table_preprocess(bit_ext_mask);

        // simd kernels
        BitExtractNetworkKernelScalar simd_nt_scalar_kernel(bit_ext_mask);
        BitExtractBlockKernelScalar   simd_bt_scalar_kernel(bit_ext_mask);
        #if defined(HAVE_SSE2)
        BitExtractNetworkKernelSSE2   simd_nt_sse2_kernel(bit_ext_mask);
        BitExtractBlockKernelSSE2     simd_bt_sse2_kernel(bit_ext_mask);
        #endif
        #if defined(HAVE_AVX2)
        BitExtractNetworkKernelAVX2   simd_nt_avx2_kernel(bit_ext_mask);
        BitExtractBlockKernelAVX2     simd_bt_avx2_kernel(bit_ext_mask);
        #endif
        #if defined(HAVE_AVX512)
        BitExtractNetworkKernelAVX512 simd_nt_avx512_kernel(bit_ext_mask);
        BitExtractBlockKernelAVX512   simd_bt_avx512_kernel(bit_ext_mask);
        #endif
        #if defined(HAVE_NEON)
        BitExtractNetworkKernelNEON   simd_nt_neon_kernel(bit_ext_mask);
        BitExtractBlockKernelNEON     simd_bt_neon_kernel(bit_ext_mask);
        #endif

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
                    return compute_spaced_kmer_hash_comin(
                        seq, k, comin_mask, comin_compute_spaced_kmer
                    );
                }
            ),
            bench(
                "comin_improved",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_comin(
                        seq, k, comin_mask, comin_compute_spaced_kmer_improved
                    );
                }
            ),

            // char_to_nt_switch
            // #if defined(HAVE_BMI2)
            // bench(
            //     "bit_extract_pext_char_to_nt_switch",
            //     [&](std::string const& seq){
            //         return compute_spaced_kmer_hash(
            //             seq, k, bit_ext_mask, char_to_nt_switch, bit_extract_pext
            //         );
            //     }
            // ),
            // #endif
            // bench(
            //     "bit_extract_bitloop_char_to_nt_switch",
            //     [&](std::string const& seq){
            //         return compute_spaced_kmer_hash(
            //             seq, k, bit_ext_mask, char_to_nt_switch, bit_extract_bitloop
            //         );
            //     }
            // ),
            // bench(
            //     "bit_extract_byte_table_char_to_nt_switch",
            //     [&](std::string const& seq){
            //         return compute_spaced_kmer_hash(
            //             seq, k, bit_ext_mask, char_to_nt_switch, bit_extract_byte_table
            //         );
            //     }
            // ),
            // bench(
            //     "bit_extract_block_table_char_to_nt_switch",
            //     [&](std::string const& seq){
            //         return compute_spaced_kmer_hash(
            //             seq, k, bit_ext_block_mask, char_to_nt_switch, bit_extract_block_table
            //         );
            //     }
            // ),

            // char_to_nt_table
            #if defined(HAVE_BMI2)
            bench(
                "bit_extract_pext_char_to_nt_table",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash(
                        seq, k, bit_ext_mask, char_to_nt_table_throw, bit_extract_pext
                    );
                }
            ),
            #endif
            bench(
                "bit_extract_bitloop_char_to_nt_table",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash(
                        seq, k, bit_ext_mask, char_to_nt_table_throw, bit_extract_bitloop
                    );
                }
            ),
            bench(
                "bit_extract_byte_table_char_to_nt_table",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash(
                        seq, k, bit_ext_mask, char_to_nt_table_throw, bit_extract_byte_table
                    );
                }
            ),
            bench(
                "bit_extract_block_table_char_to_nt_table",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash(
                        seq, k, bit_ext_block_mask, char_to_nt_table_throw, bit_extract_block_table
                    );
                }
            ),
            bench(
                "bit_extract_network_table_char_to_nt_table",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash(
                        seq, k, bit_ext_network_table, char_to_nt_table_throw, bit_extract_network_table
                    );
                }
            ),

            // simd kernels
            #if defined(HAVE_SSE2)
            bench(
                "compute_spaced_kmer_hash_simd_nt_sse2",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_simd(
                        seq, k, simd_nt_sse2_kernel
                    );
                }
            ),
            bench(
                "compute_spaced_kmer_hash_simd_bt_sse2",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_simd(
                        seq, k, simd_bt_sse2_kernel
                    );
                }
            ),
            #endif
            #if defined(HAVE_AVX2)
            bench(
                "compute_spaced_kmer_hash_simd_nt_avx2",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_simd(
                        seq, k, simd_nt_avx2_kernel
                    );
                }
            ),
            bench(
                "compute_spaced_kmer_hash_simd_bt_avx2",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_simd(
                        seq, k, simd_bt_avx2_kernel
                    );
                }
            ),
            #endif
            #if defined(HAVE_AVX512)
            bench(
                "compute_spaced_kmer_hash_simd_nt_avx512",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_simd(
                        seq, k, simd_nt_avx512_kernel
                    );
                }
            ),
            bench(
                "compute_spaced_kmer_hash_simd_bt_avx512",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_simd(
                        seq, k, simd_bt_avx512_kernel
                    );
                }
            ),
            #endif
            #if defined(HAVE_NEON)
            bench(
                "compute_spaced_kmer_hash_simd_nt_neon",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_simd(
                        seq, k, simd_nt_neon_kernel
                    );
                }
            ),
            bench(
                "compute_spaced_kmer_hash_simd_bt_neon",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_simd(
                        seq, k, simd_bt_neon_kernel
                    );
                }
            ),
            #endif
            bench(
                "compute_spaced_kmer_hash_simd_nt_scalar",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_simd(
                        seq, k, simd_nt_scalar_kernel
                    );
                }
            ),
            bench(
                "compute_spaced_kmer_hash_simd_bt_scalar",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_simd(
                        seq, k, simd_bt_scalar_kernel
                    );
                }
            )

            // char_to_nt_ascii
            // #if defined(HAVE_BMI2)
            // bench(
            //     "bit_extract_pext_char_to_nt_ascii",
            //     [&](std::string const& seq){
            //         return compute_spaced_kmer_hash(
            //             seq, k, bit_ext_mask, char_to_nt_ascii, bit_extract_pext
            //         );
            //     }
            // ),
            // #endif
            // bench(
            //     "bit_extract_bitloop_char_to_nt_ascii",
            //     [&](std::string const& seq){
            //         return compute_spaced_kmer_hash(
            //             seq, k, bit_ext_mask, char_to_nt_ascii, bit_extract_bitloop
            //         );
            //     }
            // ),
            // bench(
            //     "bit_extract_byte_table_char_to_nt_ascii",
            //     [&](std::string const& seq){
            //         return compute_spaced_kmer_hash(
            //             seq, k, bit_ext_mask, char_to_nt_ascii, bit_extract_byte_table
            //         );
            //     }
            // ),
            // bench(
            //     "bit_extract_block_table_char_to_nt_ascii",
            //     [&](std::string const& seq){
            //         return compute_spaced_kmer_hash(
            //             seq, k, bit_ext_block_mask, char_to_nt_ascii, bit_extract_block_table
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
