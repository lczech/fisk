#include <algorithm>
#include <bit>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "spaced_kmer_extract_ascii_checked_single/bench.hpp"
#include "microbench.hpp"
#include "utils.hpp"

using namespace fisk;

// Run the benchmarks of all techniques for the masks in Layout `L`, in their own suite.run(), as
// Microbench::run() cross-validates the sinks of all benches within one call, and the Layouts emit
// different values. The `baselines` are run alongside, and hence need to emit the same values.
template <Layout L, typename... Baselines>
static void bench_spaced_kmer_extract_ascii_checked_single_layout_(
    Microbench<std::string>& suite,
    std::vector<std::string> const& sequences,
    std::string const& mask,
    std::vector<std::uint64_t>& sink_buffer,
    std::ostream& csv_os,
    std::string const& suite_title,
    std::string const& case_label,
    Baselines&&... baselines
) {
    // The Layout goes into the case, next to the mask, as for the other suites.
    std::string const layout_case
        = std::string("layout=") + ((L == Layout::kMSB) ? "msb" : "lsb") + ";" + case_label;

    // Prepare masks for all implementations as needed
    auto const bit_ext_mask = make_spaced_kmer_mask<L>(mask);
    auto const bit_ext_block_mask = make_spaced_kmer_block_table(bit_ext_mask);
    auto const bit_ext_butterfly_table = make_spaced_kmer_butterfly_table(bit_ext_mask);

    // simd kernels
    auto const simd_bf_scalar_kernel
        = make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdScalar>>(bit_ext_mask);
    auto const simd_bt_scalar_kernel
        = make_spaced_kmer_kernel<BitExtractKernelBlockShift<SimdScalar>>(bit_ext_mask);
    #if defined(FISK_HAS_SSE2)
    auto const simd_bf_sse2_kernel
        = make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdSSE2>>(bit_ext_mask);
    auto const simd_bt_sse2_kernel
        = make_spaced_kmer_kernel<BitExtractKernelBlockShift<SimdSSE2>>(bit_ext_mask);
    #endif
    #if defined(FISK_HAS_AVX2)
    auto const simd_bf_avx2_kernel
        = make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdAVX2>>(bit_ext_mask);
    auto const simd_bt_avx2_kernel
        = make_spaced_kmer_kernel<BitExtractKernelBlockShift<SimdAVX2>>(bit_ext_mask);
    #endif
    #if defined(FISK_HAS_AVX512)
    auto const simd_bf_avx512_kernel
        = make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdAVX512>>(bit_ext_mask);
    auto const simd_bt_avx512_kernel
        = make_spaced_kmer_kernel<BitExtractKernelBlockShift<SimdAVX512>>(bit_ext_mask);
    #endif
    #if defined(FISK_HAS_NEON)
    auto const simd_bf_neon_kernel
        = make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdNEON>>(bit_ext_mask);
    auto const simd_bt_neon_kernel
        = make_spaced_kmer_kernel<BitExtractKernelBlockShift<SimdNEON>>(bit_ext_mask);
    #endif
    #if defined(FISK_HAS_BMI2)
    auto const simd_pext_kernel
        = make_spaced_kmer_kernel<BitExtractKernelPEXT>(bit_ext_mask);
    #endif

    // Run the benchmark for all algorithms
    auto results = suite.run(
        sequences, // vector<std::string>
        std::forward<Baselines>(baselines)...,

        // CharEncoderTable<Encoding::kACGT>
        #if defined(FISK_HAS_BMI2)
        bench(
            "rolling_scalar_pext",
            [&](std::string const& seq) {
                return run_var_rolling_scalar_pext(seq, bit_ext_mask, sink_buffer);
            }
        ),
        #endif
        bench(
            "rolling_scalar_bit_loop",
            [&](std::string const& seq) {
                return run_var_rolling_scalar_bit_loop(seq, bit_ext_mask, sink_buffer);
            }
        ),
        bench(
            "rolling_scalar_byte_lookup",
            [&](std::string const& seq) {
                return run_var_rolling_scalar_byte_lookup(seq, bit_ext_mask, sink_buffer);
            }
        ),
        bench(
            "rolling_scalar_block_shift",
            [&](std::string const& seq) {
                return run_var_rolling_scalar_block_shift(seq, bit_ext_block_mask, sink_buffer);
            }
        ),
        bench(
            "rolling_scalar_block_shift_unrolled2",
            [&](std::string const& seq) {
                return run_var_rolling_scalar_block_shift_unrolled2(
                    seq, bit_ext_block_mask, sink_buffer
                );
            }
        ),
        bench(
            "rolling_scalar_block_shift_unrolled4",
            [&](std::string const& seq) {
                return run_var_rolling_scalar_block_shift_unrolled4(
                    seq, bit_ext_block_mask, sink_buffer
                );
            }
        ),
        bench(
            "rolling_scalar_block_shift_unrolled8",
            [&](std::string const& seq) {
                return run_var_rolling_scalar_block_shift_unrolled8(
                    seq, bit_ext_block_mask, sink_buffer
                );
            }
        ),
        bench(
            "rolling_scalar_butterfly",
            [&](std::string const& seq) {
                return run_var_rolling_scalar_butterfly(
                    seq, bit_ext_butterfly_table, sink_buffer
                );
            }
        ),

        // simd kernels
        #if defined(FISK_HAS_SSE2)
        bench(
            "rolling_simd_butterfly_sse2_scalars",
            [&](std::string const& seq) {
                return run_var_rolling_simd_butterfly_sse2_scalars(
                    seq, simd_bf_sse2_kernel, sink_buffer
                );
            }
        ),
        bench(
            "rolling_simd_block_shift_sse2_scalars",
            [&](std::string const& seq) {
                return run_var_rolling_simd_block_shift_sse2_scalars(
                    seq, simd_bt_sse2_kernel, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_AVX2)
        bench(
            "rolling_simd_butterfly_avx2_scalars",
            [&](std::string const& seq) {
                return run_var_rolling_simd_butterfly_avx2_scalars(
                    seq, simd_bf_avx2_kernel, sink_buffer
                );
            }
        ),
        bench(
            "rolling_simd_block_shift_avx2_scalars",
            [&](std::string const& seq) {
                return run_var_rolling_simd_block_shift_avx2_scalars(
                    seq, simd_bt_avx2_kernel, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_AVX512)
        bench(
            "rolling_simd_butterfly_avx512_scalars",
            [&](std::string const& seq) {
                return run_var_rolling_simd_butterfly_avx512_scalars(
                    seq, simd_bf_avx512_kernel, sink_buffer
                );
            }
        ),
        bench(
            "rolling_simd_block_shift_avx512_scalars",
            [&](std::string const& seq) {
                return run_var_rolling_simd_block_shift_avx512_scalars(
                    seq, simd_bt_avx512_kernel, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_NEON)
        bench(
            "rolling_simd_butterfly_neon_scalars",
            [&](std::string const& seq) {
                return run_var_rolling_simd_butterfly_neon_scalars(
                    seq, simd_bf_neon_kernel, sink_buffer
                );
            }
        ),
        bench(
            "rolling_simd_block_shift_neon_scalars",
            [&](std::string const& seq) {
                return run_var_rolling_simd_block_shift_neon_scalars(
                    seq, simd_bt_neon_kernel, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_BMI2)
        bench(
            "rolling_simd_pext_scalars",
            [&](std::string const& seq) {
                return run_var_rolling_simd_pext_scalars(seq, simd_pext_kernel, sink_buffer);
            }
        ),
        #endif
        bench(
            "rolling_simd_butterfly_scalar_scalars",
            [&](std::string const& seq) {
                return run_var_rolling_simd_butterfly_scalar_scalars(
                    seq, simd_bf_scalar_kernel, sink_buffer
                );
            }
        ),
        bench(
            "rolling_simd_block_shift_scalar_scalars",
            [&](std::string const& seq) {
                return run_var_rolling_simd_block_shift_scalar_scalars(
                    seq, simd_bt_scalar_kernel, sink_buffer
                );
            }
        )
    );
    write_csv_rows(csv_os, suite_title, layout_case, results, kSinkName);

    // The variants that take the vectors whole, in their own suite.run(), see bench.hpp. Their sums
    // match across all archs and kernels, as the invalid lanes hold the same values in each: the
    // bits of the invalid characters, and zero past the end of the sequence.
    auto const vector_results = suite.run(
        sequences,
        #if defined(FISK_HAS_SSE2)
        bench(
            "rolling_simd_butterfly_sse2_vectors",
            [&](std::string const& seq) {
                return run_var_rolling_simd_butterfly_sse2_vectors(
                    seq, simd_bf_sse2_kernel, sink_buffer
                );
            }
        ),
        bench(
            "rolling_simd_block_shift_sse2_vectors",
            [&](std::string const& seq) {
                return run_var_rolling_simd_block_shift_sse2_vectors(
                    seq, simd_bt_sse2_kernel, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_AVX2)
        bench(
            "rolling_simd_butterfly_avx2_vectors",
            [&](std::string const& seq) {
                return run_var_rolling_simd_butterfly_avx2_vectors(
                    seq, simd_bf_avx2_kernel, sink_buffer
                );
            }
        ),
        bench(
            "rolling_simd_block_shift_avx2_vectors",
            [&](std::string const& seq) {
                return run_var_rolling_simd_block_shift_avx2_vectors(
                    seq, simd_bt_avx2_kernel, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_AVX512)
        bench(
            "rolling_simd_butterfly_avx512_vectors",
            [&](std::string const& seq) {
                return run_var_rolling_simd_butterfly_avx512_vectors(
                    seq, simd_bf_avx512_kernel, sink_buffer
                );
            }
        ),
        bench(
            "rolling_simd_block_shift_avx512_vectors",
            [&](std::string const& seq) {
                return run_var_rolling_simd_block_shift_avx512_vectors(
                    seq, simd_bt_avx512_kernel, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_NEON)
        bench(
            "rolling_simd_butterfly_neon_vectors",
            [&](std::string const& seq) {
                return run_var_rolling_simd_butterfly_neon_vectors(
                    seq, simd_bf_neon_kernel, sink_buffer
                );
            }
        ),
        bench(
            "rolling_simd_block_shift_neon_vectors",
            [&](std::string const& seq) {
                return run_var_rolling_simd_block_shift_neon_vectors(
                    seq, simd_bt_neon_kernel, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_BMI2)
        bench(
            "rolling_simd_pext_vectors",
            [&](std::string const& seq) {
                return run_var_rolling_simd_pext_vectors(seq, simd_pext_kernel, sink_buffer);
            }
        ),
        #endif
        bench(
            "rolling_simd_butterfly_scalar_vectors",
            [&](std::string const& seq) {
                return run_var_rolling_simd_butterfly_scalar_vectors(
                    seq, simd_bf_scalar_kernel, sink_buffer
                );
            }
        ),
        bench(
            "rolling_simd_block_shift_scalar_vectors",
            [&](std::string const& seq) {
                return run_var_rolling_simd_block_shift_scalar_vectors(
                    seq, simd_bt_scalar_kernel, sink_buffer
                );
            }
        )
    );
    write_csv_rows(csv_os, suite_title, layout_case, vector_results, kSinkName);
}

void bench_spaced_kmer_extract_ascii_checked_single(
    std::vector<std::string> const& sequences,
    std::vector<std::string> const& masks,
    std::ostream& csv_os
) {
    std::size_t const rounds = 1;
    std::size_t const repeats = 8;

    // User output
    std::string const suite_title = "spaced_kmer_extract_ascii_checked_single";
    std::cout << "\n=== spaced k-mer extract single mask ===\n";
    std::cout << "rounds=" << rounds << ", repeats=" << repeats << "\n";

    write_csv_header(csv_os);

    // Backing storage for the Write sink (see sink.hpp), sized to the longest sequence
    // actually benchmarked (rounded up to a power of two) so that no call here ever wraps
    // it. Different techniques here could in principle differ in how many spaced k-mers
    // they actually emit (e.g. one skipping windows a differently-behaved one still emits
    // for), which would make a wrapped buffer's leftover contents depend on which technique
    // last touched each slot; never wrapping sidesteps that regardless of whether it would
    // in fact occur. The `_vectors` variants also write the lanes of their last vector past the
    // end of the sequence, at most 8.
    std::size_t max_seq_len = 0;
    for (auto const& seq : sequences) {
        max_seq_len = std::max(max_seq_len, seq.size());
    }
    std::vector<std::uint64_t> sink_buffer(std::bit_ceil(max_seq_len + 8), 0);

    // Run a benchmark for each mask
    for (std::size_t m = 0; m < masks.size(); ++m) {
        auto const k = masks[m].size();
        if (stdout_is_terminal()) {
            std::cout << "\rmask ";
            std::cout << std::setw(2) << (m + 1) << " / " << masks.size() << "\n";
        }

        // Masks for the baselines; the others are prepared per Layout.
        auto const naive_mask = spaced_kmer_mask_positions(masks[m]);

        // Prepare a benchmark with repititions
        Microbench<std::string> suite(suite_title);
        suite
            .rounds(rounds)
            .repeats(repeats)
            .units_fn([k](std::string const& seq) {
                // 1 unit per original (unspaced) k-mer
                return static_cast<double>(seq.size() - k + 1);
            });

        // Run the benchmarks in both Layouts, with the baselines alongside MSB, which they match.
        std::string const case_label = "mask=" + std::to_string(m);
        bench_spaced_kmer_extract_ascii_checked_single_layout_<Layout::kMSB>(
            suite, sequences, masks[m], sink_buffer, csv_os, suite_title, case_label,

            // naive, as baseline and validity check
            bench(
                "missh",
                [&, k](std::string const& seq) {
                    return run_var_missh(seq, k, naive_mask, sink_buffer);
                }
            ),
            bench(
                "naive",
                [&, k](std::string const& seq) {
                    return run_var_naive(seq, k, naive_mask, sink_buffer);
                }
            )
        );
        bench_spaced_kmer_extract_ascii_checked_single_layout_<Layout::kLSB>(
            suite, sequences, masks[m], sink_buffer, csv_os, suite_title, case_label
        );
    }
    if (stdout_is_terminal()) {
        std::cout << "\n";
    }
}
