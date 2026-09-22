#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fisk/kmer_spaced/selector.hpp"
#include "kmer_spaced_multi/bench.hpp"
#include "microbench.hpp"
#include "utils.hpp"

using namespace fisk;

void bench_kmer_spaced_multi(
    std::vector<std::string> const& sequences,
    std::vector<std::vector<std::string>> const& multi_masks,
    std::ostream& csv_os
) {
    // Check masks for consistent value of k and mask characters
    for (auto const& masks : multi_masks) {
        if (masks.size() == 0) {
            throw std::invalid_argument("Mask set is empty");
        }
        std::size_t k = masks[0].size();
        for (auto const& mask : masks) {
            if (mask.size() != k) {
                throw std::invalid_argument("Inconsistent mask lengths in mask set");
            }
        }
    }

    std::size_t const rounds = 1;
    std::size_t const repeats = 4;

    // User output
    std::string const suite_title = "kmer_spaced_multi";
    std::cout << "\n=== spaced k-mer extract multi masks ===\n";
    std::cout << "rounds=" << rounds << ", repeats=" << repeats << "\n";

    write_csv_header(csv_os);

    // Run a benchmark for each mask set
    for (std::size_t m = 0; m < multi_masks.size(); ++m) {
        if (stdout_is_terminal()) {
            std::cout << "\rmask set ";
            std::cout << std::setw(2) << (m + 1) << " / " << multi_masks.size() << "\n";
        }
        auto const k = multi_masks[m][0].size();

        // Prepare masks for all implementations as needed
        std::vector<std::uint64_t> raw_masks;
        std::vector<std::vector<std::size_t>> naive_masks;
        std::vector<BitExtractMask> bit_ext_masks;
        std::vector<BitExtractBlockTable> bit_ext_block_masks;
        std::vector<BitExtractButterflyTable> bit_ext_butterfly_tables;
        for (auto const& mask : multi_masks[m]) {
            raw_masks.push_back(prepare_spaced_kmer_bit_extract_mask(mask));
            naive_masks.push_back(prepare_naive_mask(mask));
            bit_ext_masks.push_back(BitExtractMask(raw_masks.back()));
            bit_ext_block_masks.push_back(bit_extract_block_table_preprocess(raw_masks.back()));
            bit_ext_butterfly_tables.push_back(bit_extract_butterfly_table_preprocess(raw_masks.back()));

            // Just to test the selector implementation
            std::cout << "fastest mode: ";
            std::cout << spaced_kmer_mode_name(spaced_kmer_selector(raw_masks.back(), k)) << "\n";
        }

        // simd kernels
        BitExtractKernelDispatcher<BitExtractKernelButterflyScalar> simd_bf_scalar_kernel(raw_masks);
        BitExtractKernelDispatcher<BitExtractKernelBlockScalar<>>   simd_bt_scalar_kernel(raw_masks);
        #if defined(FISK_HAS_SSE2)
        BitExtractKernelDispatcher<BitExtractKernelButterflySSE2>   simd_bf_sse2_kernel(raw_masks);
        BitExtractKernelDispatcher<BitExtractKernelBlockSSE2<>>     simd_bt_sse2_kernel(raw_masks);
        #endif
        #if defined(FISK_HAS_AVX2)
        BitExtractKernelDispatcher<BitExtractKernelButterflyAVX2>   simd_bf_avx2_kernel(raw_masks);
        BitExtractKernelDispatcher<BitExtractKernelBlockAVX2<>>     simd_bt_avx2_kernel(raw_masks);
        #endif
        #if defined(FISK_HAS_AVX512)
        BitExtractKernelDispatcher<BitExtractKernelButterflyAVX512> simd_bf_avx512_kernel(raw_masks);
        BitExtractKernelDispatcher<BitExtractKernelBlockAVX512<>>   simd_bt_avx512_kernel(raw_masks);
        #endif
        #if defined(FISK_HAS_NEON)
        BitExtractKernelDispatcher<BitExtractKernelButterflyNEON>   simd_bf_neon_kernel(raw_masks);
        BitExtractKernelDispatcher<BitExtractKernelBlockNEON<>>     simd_bt_neon_kernel(raw_masks);
        #endif
        #if defined(FISK_HAS_BMI2)
        BitExtractKernelDispatcher<BitExtractKernelPEXT<>>          simd_pext_kernel(raw_masks);
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
                [&, k](std::string const& seq) {
                    return run_var_naive(seq, k, naive_masks);
                }
            ),

            // CharEncoderTable<Encoding::kACGT>
            #if defined(FISK_HAS_BMI2)
            bench(
                "pext",
                [&, k](std::string const& seq) {
                    return run_var_pext(seq, k, bit_ext_masks);
                }
            ),
            #endif
            bench(
                "bitloop",
                [&, k](std::string const& seq) {
                    return run_var_bitloop(seq, k, bit_ext_masks);
                }
            ),
            bench(
                "byte_table",
                [&, k](std::string const& seq) {
                    return run_var_byte_table(seq, k, bit_ext_masks);
                }
            ),
            bench(
                "block_table",
                [&, k](std::string const& seq) {
                    return run_var_block_table(seq, k, bit_ext_block_masks);
                }
            ),
            bench(
                "block_table_unrolled2",
                [&, k](std::string const& seq) {
                    return run_var_block_table_unrolled2(seq, k, bit_ext_block_masks);
                }
            ),
            bench(
                "block_table_unrolled4",
                [&, k](std::string const& seq) {
                    return run_var_block_table_unrolled4(seq, k, bit_ext_block_masks);
                }
            ),
            bench(
                "block_table_unrolled8",
                [&, k](std::string const& seq) {
                    return run_var_block_table_unrolled8(seq, k, bit_ext_block_masks);
                }
            ),
            bench(
                "butterfly_table",
                [&, k](std::string const& seq) {
                    return run_var_butterfly_table(seq, k, bit_ext_butterfly_tables);
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

        std::string case_label = "mask_set=" + std::to_string(m);
        write_csv_rows(csv_os, suite_title, case_label, results);
    }
    if (stdout_is_terminal()) {
        std::cout << "\n";
    }
}
