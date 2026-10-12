#include <algorithm>
#include <bit>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fisk/kmer_spaced/selector.hpp"
#include "kmer_spaced_multi/bench.hpp"
#include "microbench.hpp"
#include "utils.hpp"

using namespace fisk;

// Run the benchmarks of all techniques for the masks in Layout `L`, in their own suite.run(), as
// Microbench::run() cross-validates the sinks of all benches within one call, and the Layouts emit
// different values. The `baselines` are run alongside, and hence need to emit the same values.
template <Layout L, typename... Baselines>
static void bench_kmer_spaced_multi_layout_(
    Microbench<std::string>& suite,
    std::vector<std::string> const& sequences,
    std::vector<std::string> const& masks,
    std::vector<std::uint64_t>& sink_buffer,
    std::ostream& csv_os,
    std::string const& suite_title,
    std::string const& case_label,
    Baselines&&... baselines
) {
    std::string const prefix = (L == Layout::kMSB) ? "msb" : "lsb";

    // Benchmark names are string views, so their strings need to outlive the run.
    std::deque<std::string> names;
    auto const name = [&](std::string_view technique) -> std::string_view {
        return names.emplace_back(prefix + "_" + std::string(technique));
    };

    // Prepare masks for all implementations as needed
    auto const bit_ext_masks = make_spaced_kmer_masks<L>(masks);
    auto const bit_ext_block_masks = make_spaced_kmer_block_tables(bit_ext_masks);
    auto const bit_ext_butterfly_tables = make_spaced_kmer_butterfly_tables(bit_ext_masks);

    // Just to test the selector implementation
    std::cout << "fastest mode " << prefix << ": ";
    std::cout << spaced_kmer_mode_name(spaced_kmer_selector(bit_ext_masks)) << "\n";

    // simd kernels
    BitExtractKernelDispatcher const simd_bf_scalar_kernel(
        make_spaced_kmer_kernels<BitExtractKernelButterfly<SimdScalar>>(bit_ext_masks)
    );
    BitExtractKernelDispatcher const simd_bt_scalar_kernel(
        make_spaced_kmer_kernels<BitExtractKernelBlock<SimdScalar>>(bit_ext_masks)
    );
    #if defined(FISK_HAS_SSE2)
    BitExtractKernelDispatcher const simd_bf_sse2_kernel(
        make_spaced_kmer_kernels<BitExtractKernelButterfly<SimdSSE2>>(bit_ext_masks)
    );
    BitExtractKernelDispatcher const simd_bt_sse2_kernel(
        make_spaced_kmer_kernels<BitExtractKernelBlock<SimdSSE2>>(bit_ext_masks)
    );
    #endif
    #if defined(FISK_HAS_AVX2)
    BitExtractKernelDispatcher const simd_bf_avx2_kernel(
        make_spaced_kmer_kernels<BitExtractKernelButterfly<SimdAVX2>>(bit_ext_masks)
    );
    BitExtractKernelDispatcher const simd_bt_avx2_kernel(
        make_spaced_kmer_kernels<BitExtractKernelBlock<SimdAVX2>>(bit_ext_masks)
    );
    #endif
    #if defined(FISK_HAS_AVX512)
    BitExtractKernelDispatcher const simd_bf_avx512_kernel(
        make_spaced_kmer_kernels<BitExtractKernelButterfly<SimdAVX512>>(bit_ext_masks)
    );
    BitExtractKernelDispatcher const simd_bt_avx512_kernel(
        make_spaced_kmer_kernels<BitExtractKernelBlock<SimdAVX512>>(bit_ext_masks)
    );
    #endif
    #if defined(FISK_HAS_NEON)
    BitExtractKernelDispatcher const simd_bf_neon_kernel(
        make_spaced_kmer_kernels<BitExtractKernelButterfly<SimdNEON>>(bit_ext_masks)
    );
    BitExtractKernelDispatcher const simd_bt_neon_kernel(
        make_spaced_kmer_kernels<BitExtractKernelBlock<SimdNEON>>(bit_ext_masks)
    );
    #endif
    #if defined(FISK_HAS_BMI2)
    BitExtractKernelDispatcher const simd_pext_kernel(
        make_spaced_kmer_kernels<BitExtractKernelPEXT<>>(bit_ext_masks)
    );
    #endif

    // Run the benchmark for all algorithms
    auto results = suite.run(
        sequences, // vector<std::string>
        std::forward<Baselines>(baselines)...,

        // CharEncoderTable<Encoding::kACGT>
        #if defined(FISK_HAS_BMI2)
        bench(
            name("pext"),
            [&](std::string const& seq) {
                return run_var_pext(seq, bit_ext_masks, sink_buffer);
            }
        ),
        #endif
        bench(
            name("bitloop"),
            [&](std::string const& seq) {
                return run_var_bitloop(seq, bit_ext_masks, sink_buffer);
            }
        ),
        bench(
            name("byte_table"),
            [&](std::string const& seq) {
                return run_var_byte_table(seq, bit_ext_masks, sink_buffer);
            }
        ),
        bench(
            name("block_table"),
            [&](std::string const& seq) {
                return run_var_block_table(seq, bit_ext_block_masks, sink_buffer);
            }
        ),
        bench(
            name("block_table_unrolled2"),
            [&](std::string const& seq) {
                return run_var_block_table_unrolled2(seq, bit_ext_block_masks, sink_buffer);
            }
        ),
        bench(
            name("block_table_unrolled4"),
            [&](std::string const& seq) {
                return run_var_block_table_unrolled4(seq, bit_ext_block_masks, sink_buffer);
            }
        ),
        bench(
            name("block_table_unrolled8"),
            [&](std::string const& seq) {
                return run_var_block_table_unrolled8(seq, bit_ext_block_masks, sink_buffer);
            }
        ),
        bench(
            name("butterfly_table"),
            [&](std::string const& seq) {
                return run_var_butterfly_table(seq, bit_ext_butterfly_tables, sink_buffer);
            }
        ),

        // simd kernels
        #if defined(FISK_HAS_SSE2)
        bench(
            name("simd_butterfly_table_sse2_by_mask"),
            [&](std::string const& seq) {
                return run_var_simd_butterfly_table_sse2_by_mask(
                    seq, simd_bf_sse2_kernel, sink_buffer
                );
            }
        ),
        bench(
            name("simd_butterfly_table_sse2_by_position"),
            [&](std::string const& seq) {
                return run_var_simd_butterfly_table_sse2_by_position(
                    seq, simd_bf_sse2_kernel, sink_buffer
                );
            }
        ),
        bench(
            name("simd_block_table_sse2_by_mask"),
            [&](std::string const& seq) {
                return run_var_simd_block_table_sse2_by_mask(
                    seq, simd_bt_sse2_kernel, sink_buffer
                );
            }
        ),
        bench(
            name("simd_block_table_sse2_by_position"),
            [&](std::string const& seq) {
                return run_var_simd_block_table_sse2_by_position(
                    seq, simd_bt_sse2_kernel, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_AVX2)
        bench(
            name("simd_butterfly_table_avx2_by_mask"),
            [&](std::string const& seq) {
                return run_var_simd_butterfly_table_avx2_by_mask(
                    seq, simd_bf_avx2_kernel, sink_buffer
                );
            }
        ),
        bench(
            name("simd_butterfly_table_avx2_by_position"),
            [&](std::string const& seq) {
                return run_var_simd_butterfly_table_avx2_by_position(
                    seq, simd_bf_avx2_kernel, sink_buffer
                );
            }
        ),
        bench(
            name("simd_block_table_avx2_by_mask"),
            [&](std::string const& seq) {
                return run_var_simd_block_table_avx2_by_mask(
                    seq, simd_bt_avx2_kernel, sink_buffer
                );
            }
        ),
        bench(
            name("simd_block_table_avx2_by_position"),
            [&](std::string const& seq) {
                return run_var_simd_block_table_avx2_by_position(
                    seq, simd_bt_avx2_kernel, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_AVX512)
        bench(
            name("simd_butterfly_table_avx512_by_mask"),
            [&](std::string const& seq) {
                return run_var_simd_butterfly_table_avx512_by_mask(
                    seq, simd_bf_avx512_kernel, sink_buffer
                );
            }
        ),
        bench(
            name("simd_butterfly_table_avx512_by_position"),
            [&](std::string const& seq) {
                return run_var_simd_butterfly_table_avx512_by_position(
                    seq, simd_bf_avx512_kernel, sink_buffer
                );
            }
        ),
        bench(
            name("simd_block_table_avx512_by_mask"),
            [&](std::string const& seq) {
                return run_var_simd_block_table_avx512_by_mask(
                    seq, simd_bt_avx512_kernel, sink_buffer
                );
            }
        ),
        bench(
            name("simd_block_table_avx512_by_position"),
            [&](std::string const& seq) {
                return run_var_simd_block_table_avx512_by_position(
                    seq, simd_bt_avx512_kernel, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_NEON)
        bench(
            name("simd_butterfly_table_neon_by_mask"),
            [&](std::string const& seq) {
                return run_var_simd_butterfly_table_neon_by_mask(
                    seq, simd_bf_neon_kernel, sink_buffer
                );
            }
        ),
        bench(
            name("simd_butterfly_table_neon_by_position"),
            [&](std::string const& seq) {
                return run_var_simd_butterfly_table_neon_by_position(
                    seq, simd_bf_neon_kernel, sink_buffer
                );
            }
        ),
        bench(
            name("simd_block_table_neon_by_mask"),
            [&](std::string const& seq) {
                return run_var_simd_block_table_neon_by_mask(
                    seq, simd_bt_neon_kernel, sink_buffer
                );
            }
        ),
        bench(
            name("simd_block_table_neon_by_position"),
            [&](std::string const& seq) {
                return run_var_simd_block_table_neon_by_position(
                    seq, simd_bt_neon_kernel, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_BMI2)
        bench(
            name("simd_pext_by_mask"),
            [&](std::string const& seq) {
                return run_var_simd_pext_by_mask(seq, simd_pext_kernel, sink_buffer);
            }
        ),
        bench(
            name("simd_pext_by_position"),
            [&](std::string const& seq) {
                return run_var_simd_pext_by_position(seq, simd_pext_kernel, sink_buffer);
            }
        ),
        #endif
        bench(
            name("simd_butterfly_table_scalar_by_mask"),
            [&](std::string const& seq) {
                return run_var_simd_butterfly_table_scalar_by_mask(
                    seq, simd_bf_scalar_kernel, sink_buffer
                );
            }
        ),
        bench(
            name("simd_butterfly_table_scalar_by_position"),
            [&](std::string const& seq) {
                return run_var_simd_butterfly_table_scalar_by_position(
                    seq, simd_bf_scalar_kernel, sink_buffer
                );
            }
        ),
        bench(
            name("simd_block_table_scalar_by_mask"),
            [&](std::string const& seq) {
                return run_var_simd_block_table_scalar_by_mask(
                    seq, simd_bt_scalar_kernel, sink_buffer
                );
            }
        ),
        bench(
            name("simd_block_table_scalar_by_position"),
            [&](std::string const& seq) {
                return run_var_simd_block_table_scalar_by_position(
                    seq, simd_bt_scalar_kernel, sink_buffer
                );
            }
        )
    );
    write_csv_rows(csv_os, suite_title, case_label, results, kSinkName);
}

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

    // Backing storage for the Write sink (see sink.hpp), sized to the longest sequence actually
    // benchmarked, times the largest mask set (every position is touched once per mask in the
    // set), rounded up to a power of two so that no call here ever wraps it. Different techniques
    // here could in principle differ in how many spaced k-mers they actually emit (e.g. one
    // skipping windows a differently-behaved one still emits for), which would make a wrapped
    // buffer's leftover contents depend on which technique last touched each slot; never wrapping
    // sidesteps that regardless of whether it would in fact occur.
    std::size_t max_seq_len = 0;
    for (auto const& seq : sequences) {
        max_seq_len = std::max(max_seq_len, seq.size());
    }
    std::size_t max_masks = 1;
    for (auto const& masks : multi_masks) {
        max_masks = std::max(max_masks, masks.size());
    }
    std::vector<std::uint64_t> sink_buffer(
        std::bit_ceil(std::max<std::size_t>(max_seq_len, 1) * max_masks), 0
    );

    // Run a benchmark for each mask set
    for (std::size_t m = 0; m < multi_masks.size(); ++m) {
        if (stdout_is_terminal()) {
            std::cout << "\rmask set ";
            std::cout << std::setw(2) << (m + 1) << " / " << multi_masks.size() << "\n";
        }
        auto const k = multi_masks[m][0].size();

        // Masks for the baselines; the others are prepared per Layout.
        std::vector<std::vector<std::size_t>> naive_masks;
        for (auto const& mask : multi_masks[m]) {
            naive_masks.push_back(spaced_kmer_mask_positions(mask));
        }

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

        // Run the benchmarks in both Layouts, with the baselines alongside MSB, which they match.
        std::string const case_label = "mask_set=" + std::to_string(m);
        bench_kmer_spaced_multi_layout_<Layout::kMSB>(
            suite, sequences, multi_masks[m], sink_buffer, csv_os, suite_title, case_label,

            // naive, as baseline and validity check
            bench(
                "naive",
                [&, k](std::string const& seq) {
                    return run_var_naive(seq, k, naive_masks, sink_buffer);
                }
            )
        );
        bench_kmer_spaced_multi_layout_<Layout::kLSB>(
            suite, sequences, multi_masks[m], sink_buffer, csv_os, suite_title, case_label
        );
    }
    if (stdout_is_terminal()) {
        std::cout << "\n";
    }
}
