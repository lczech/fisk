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
inline void bench_kmer_spaced_multi(
    std::vector<std::string> const& sequences,
    std::vector<std::vector<std::string>> const& multi_masks,
    std::ostream& csv_os
) {
    // Check masks for consistent value of k and mask characters
    for( auto const& masks : multi_masks ) {
        if( masks.size() == 0 ) {
            throw std::invalid_argument( "Mask set is empty" );
        }
        size_t k = masks[0].size();
        for( auto const& mask : masks ) {
            if( mask.size() != k ) {
                throw std::invalid_argument( "Inconsistent mask lengths in mask set" );
            }
        }
    }

    std::size_t const rounds = 1;
    std::size_t const repeats = 4;

    // User output
    std::string const suite_title = "kmer_spaced_multi";
    std::cout << "\n=== spaced k-mer extract multi masks ===\n";
    std::cout << "rounds=" << rounds << ", repeats=" << repeats << "\n";

    // Prepare csv output file with benchmark results
    write_csv_header(csv_os);

    // Run a benchmark for each mask
    for( size_t m = 0; m < multi_masks.size(); ++m) {
        if( stdout_is_terminal() ) {
            std::cout << "\rmask set "
                << std::setw(2) << (m+1) << " / " << multi_masks.size()
                << std::flush;
        }
        auto const k = multi_masks[m][0].size();

        // Prepare masks for all implementations as needed
        std::vector<std::uint64_t> raw_masks;
        std::vector<std::vector<size_t>> naive_masks;
        std::vector<BitExtractMask> bit_ext_masks;
        std::vector<BitExtractBlockTable> bit_ext_block_masks;
        std::vector<BitExtractButterflyTable> bit_ext_butterfly_tables;
        for( auto const& mask : multi_masks[m] ) {
            raw_masks.push_back(
                prepare_spaced_kmer_bit_extract_mask(mask)
            );
            naive_masks.push_back(
                prepare_naive_mask(mask)
            );
            bit_ext_masks.push_back(
                BitExtractMask(raw_masks.back())
            );
            bit_ext_block_masks.push_back(
                bit_extract_block_table_preprocess(raw_masks.back())
            );
            bit_ext_butterfly_tables.push_back(
                bit_extract_butterfly_table_preprocess(raw_masks.back())
            );
        }

        // simd kernels
        BitExtractKernelDispatcher<BitExtractKernelButterflyScalar> simd_bf_scalar_kernel(raw_masks);
        BitExtractKernelDispatcher<BitExtractKernelBlockScalar<>>   simd_bt_scalar_kernel(raw_masks);
        #if defined(HAVE_SSE2)
        BitExtractKernelDispatcher<BitExtractKernelButterflySSE2>   simd_bf_sse2_kernel(raw_masks);
        BitExtractKernelDispatcher<BitExtractKernelBlockSSE2<>>     simd_bt_sse2_kernel(raw_masks);
        #endif
        #if defined(HAVE_AVX2)
        BitExtractKernelDispatcher<BitExtractKernelButterflyAVX2>   simd_bf_avx2_kernel(raw_masks);
        BitExtractKernelDispatcher<BitExtractKernelBlockAVX2<>>     simd_bt_avx2_kernel(raw_masks);
        #endif
        #if defined(HAVE_AVX512)
        BitExtractKernelDispatcher<BitExtractKernelButterflyAVX512> simd_bf_avx512_kernel(raw_masks);
        BitExtractKernelDispatcher<BitExtractKernelBlockAVX512<>>   simd_bt_avx512_kernel(raw_masks);
        #endif
        #if defined(HAVE_NEON)
        BitExtractKernelDispatcher<BitExtractKernelButterflyNEON>   simd_bf_neon_kernel(raw_masks);
        BitExtractKernelDispatcher<BitExtractKernelBlockNEON<>>     simd_bt_neon_kernel(raw_masks);
        #endif

        // Prepare a benchmark with repititions
        Microbench<std::string> suite(suite_title);
        suite
            .rounds(rounds)
            .repeats(repeats)
            .units_fn([&, k](std::string const& seq) {
                // 1 unit per original (unspaced) k-mer and per mask
                auto const kmer_count = static_cast<double>(seq.size() - k + 1);
                auto const mask_count = static_cast<double>(multi_masks[m].size());
                return kmer_count * mask_count;
            });

        // Run the benchmark for all algorithms
        auto results = suite.run(
            sequences, // vector<std::string>

            // naive, as baseline and validity check
            bench(
                "naive",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_naive(
                        seq, k, naive_masks, compute_spaced_kmer_naive
                    );
                }
            ),

            // char_to_nt_table
            #if defined(HAVE_BMI2)
            bench(
                "bit_extract_pext",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash(
                        seq, k, bit_ext_masks, char_to_nt_table, bit_extract_pext
                    );
                }
            ),
            #endif
            bench(
                "bit_extract_bitloop",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash(
                        seq, k, bit_ext_masks, char_to_nt_table, bit_extract_bitloop
                    );
                }
            ),
            bench(
                "bit_extract_byte_table",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash(
                        seq, k, bit_ext_masks, char_to_nt_table, bit_extract_byte_table
                    );
                }
            ),
            bench(
                "bit_extract_block_table",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash(
                        seq, k, bit_ext_block_masks, char_to_nt_table, bit_extract_block_table
                    );
                }
            ),
            bench(
                "bit_extract_block_table_unrolled2",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash(
                        seq, k, bit_ext_block_masks, char_to_nt_table, bit_extract_block_table_unrolled<2>
                    );
                }
            ),
            bench(
                "bit_extract_block_table_unrolled4",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash(
                        seq, k, bit_ext_block_masks, char_to_nt_table, bit_extract_block_table_unrolled<4>
                    );
                }
            ),
            bench(
                "bit_extract_block_table_unrolled8",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash(
                        seq, k, bit_ext_block_masks, char_to_nt_table, bit_extract_block_table_unrolled<8>
                    );
                }
            ),
            bench(
                "bit_extract_butterfly_table",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash(
                        seq, k, bit_ext_butterfly_tables, char_to_nt_table, bit_extract_butterfly_table
                    );
                }
            ),

            // simd kernels
            #if defined(HAVE_SSE2)
            bench(
                "compute_spaced_kmer_hash_simd_bf_sse2",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_simd(
                        seq, k, simd_bf_sse2_kernel
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
                "compute_spaced_kmer_hash_simd_bf_avx2",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_simd(
                        seq, k, simd_bf_avx2_kernel
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
                "compute_spaced_kmer_hash_simd_bf_avx512",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_simd(
                        seq, k, simd_bf_avx512_kernel
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
                "compute_spaced_kmer_hash_simd_bf_neon",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_simd(
                        seq, k, simd_bf_neon_kernel
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
                "compute_spaced_kmer_hash_simd_bf_scalar",
                [&](std::string const& seq){
                    return compute_spaced_kmer_hash_simd(
                        seq, k, simd_bf_scalar_kernel
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
        );

        std::string case_label = "mask_set=" + std::to_string(m);
        write_csv_rows(csv_os, suite_title, case_label, results);
    }
    if( stdout_is_terminal() ) {
        std::cout << "\n";
    }
}
