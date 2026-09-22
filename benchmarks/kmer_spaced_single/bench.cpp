#include <iostream>
#include <string>
#include <vector>

#include "fisk/kmer_spaced/selector.hpp"
#include "kmer_spaced_single/bench.hpp"
#include "microbench.hpp"
#include "utils.hpp"

using namespace fisk;

void bench_kmer_spaced_single(
    std::vector<std::string> const& sequences,
    std::vector<std::string> const& masks,
    std::ostream& csv_os
) {
    std::size_t const rounds = 1;
    std::size_t const repeats = 8;

    // User output
    std::string const suite_title = "kmer_spaced_single";
    std::cout << "\n=== spaced k-mer extract single mask ===\n";
    std::cout << "rounds=" << rounds << ", repeats=" << repeats << "\n";

    write_csv_header(csv_os);

    // Run a benchmark for each mask
    for (std::size_t m = 0; m < masks.size(); ++m) {
        auto const k = masks[m].size();
        if (stdout_is_terminal()) {
            std::cout << "\rmask ";
            std::cout << std::setw(2) << (m + 1) << " / " << masks.size() << "\n";
        }

        // Prepare masks for all implementations as needed
        auto const naive_mask = prepare_naive_mask(masks[m]);
        auto const raw_mask = prepare_spaced_kmer_bit_extract_mask(masks[m]);
        BitExtractMask const bit_ext_mask(raw_mask);
        auto const bit_ext_block_mask = bit_extract_block_table_preprocess(raw_mask);
        auto const bit_ext_butterfly_table = bit_extract_butterfly_table_preprocess(raw_mask);

        // Just to test the selector
        std::cout << "fastest mode: ";
        std::cout << spaced_kmer_mode_name(spaced_kmer_selector(raw_mask, k)) << "\n";

        // simd kernels
        BitExtractKernelButterflyScalar simd_bf_scalar_kernel(raw_mask);
        BitExtractKernelBlockScalar<>   simd_bt_scalar_kernel(raw_mask);
        #if defined(FISK_HAS_SSE2)
        BitExtractKernelButterflySSE2   simd_bf_sse2_kernel(raw_mask);
        BitExtractKernelBlockSSE2<>     simd_bt_sse2_kernel(raw_mask);
        #endif
        #if defined(FISK_HAS_AVX2)
        BitExtractKernelButterflyAVX2   simd_bf_avx2_kernel(raw_mask);
        BitExtractKernelBlockAVX2<>     simd_bt_avx2_kernel(raw_mask);
        #endif
        #if defined(FISK_HAS_AVX512)
        BitExtractKernelButterflyAVX512 simd_bf_avx512_kernel(raw_mask);
        BitExtractKernelBlockAVX512<>   simd_bt_avx512_kernel(raw_mask);
        #endif
        #if defined(FISK_HAS_NEON)
        BitExtractKernelButterflyNEON   simd_bf_neon_kernel(raw_mask);
        BitExtractKernelBlockNEON<>     simd_bt_neon_kernel(raw_mask);
        #endif
        #if defined(FISK_HAS_BMI2)
        BitExtractKernelPEXT<>          simd_pext_kernel(raw_mask);
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
                [&, k](std::string const& seq) {
                    return run_var_missh(seq, k, naive_mask);
                }
            ),
            bench(
                "naive",
                [&, k](std::string const& seq) {
                    return run_var_naive(seq, k, naive_mask);
                }
            ),

            // CharEncoderTable<Encoding::kACGT>
            #if defined(FISK_HAS_BMI2)
            bench(
                "pext",
                [&, k](std::string const& seq) {
                    return run_var_pext(seq, k, bit_ext_mask);
                }
            ),
            #endif
            bench(
                "bitloop",
                [&, k](std::string const& seq) {
                    return run_var_bitloop(seq, k, bit_ext_mask);
                }
            ),
            bench(
                "byte_table",
                [&, k](std::string const& seq) {
                    return run_var_byte_table(seq, k, bit_ext_mask);
                }
            ),
            bench(
                "block_table",
                [&, k](std::string const& seq) {
                    return run_var_block_table(seq, k, bit_ext_block_mask);
                }
            ),
            bench(
                "block_table_unrolled2",
                [&, k](std::string const& seq) {
                    return run_var_block_table_unrolled2(seq, k, bit_ext_block_mask);
                }
            ),
            bench(
                "block_table_unrolled4",
                [&, k](std::string const& seq) {
                    return run_var_block_table_unrolled4(seq, k, bit_ext_block_mask);
                }
            ),
            bench(
                "block_table_unrolled8",
                [&, k](std::string const& seq) {
                    return run_var_block_table_unrolled8(seq, k, bit_ext_block_mask);
                }
            ),
            bench(
                "butterfly_table",
                [&, k](std::string const& seq) {
                    return run_var_butterfly_table(seq, k, bit_ext_butterfly_table);
                }
            ),

            // simd kernels
            #if defined(FISK_HAS_SSE2)
            bench(
                "simd_butterfly_table_sse2",
                [&, k](std::string const& seq) {
                    return run_var_simd_butterfly_table_sse2(seq, k, simd_bf_sse2_kernel);
                }
            ),
            bench(
                "simd_block_table_sse2",
                [&, k](std::string const& seq) {
                    return run_var_simd_block_table_sse2(seq, k, simd_bt_sse2_kernel);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX2)
            bench(
                "simd_butterfly_table_avx2",
                [&, k](std::string const& seq) {
                    return run_var_simd_butterfly_table_avx2(seq, k, simd_bf_avx2_kernel);
                }
            ),
            bench(
                "simd_block_table_avx2",
                [&, k](std::string const& seq) {
                    return run_var_simd_block_table_avx2(seq, k, simd_bt_avx2_kernel);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX512)
            bench(
                "simd_butterfly_table_avx512",
                [&, k](std::string const& seq) {
                    return run_var_simd_butterfly_table_avx512(seq, k, simd_bf_avx512_kernel);
                }
            ),
            bench(
                "simd_block_table_avx512",
                [&, k](std::string const& seq) {
                    return run_var_simd_block_table_avx512(seq, k, simd_bt_avx512_kernel);
                }
            ),
            #endif
            #if defined(FISK_HAS_NEON)
            bench(
                "simd_butterfly_table_neon",
                [&, k](std::string const& seq) {
                    return run_var_simd_butterfly_table_neon(seq, k, simd_bf_neon_kernel);
                }
            ),
            bench(
                "simd_block_table_neon",
                [&, k](std::string const& seq) {
                    return run_var_simd_block_table_neon(seq, k, simd_bt_neon_kernel);
                }
            ),
            #endif
            #if defined(FISK_HAS_BMI2)
            bench(
                "simd_pext",
                [&, k](std::string const& seq) {
                    return run_var_simd_pext(seq, k, simd_pext_kernel);
                }
            ),
            #endif
            bench(
                "simd_butterfly_table_scalar",
                [&, k](std::string const& seq) {
                    return run_var_simd_butterfly_table_scalar(seq, k, simd_bf_scalar_kernel);
                }
            ),
            bench(
                "simd_block_table_scalar",
                [&, k](std::string const& seq) {
                    return run_var_simd_block_table_scalar(seq, k, simd_bt_scalar_kernel);
                }
            )
        );

        std::string case_label = "mask=" + std::to_string(m);
        write_csv_rows(csv_os, suite_title, case_label, results);
    }
    if (stdout_is_terminal()) {
        std::cout << "\n";
    }
}
