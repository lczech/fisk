#include <algorithm>
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
    auto const bit_ext_block_mask = make_spaced_kmer_block_shift_mask(bit_ext_mask);
    auto const bit_ext_butterfly_mask = make_spaced_kmer_butterfly_mask(bit_ext_mask);

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
        = make_spaced_kmer_kernel<BitExtractKernelPext>(bit_ext_mask);
    #endif

    // Scratch of the chunked variants, one each.
    #if defined(FISK_HAS_BMI2)
    ChunkedCheckedScratch<L> scratch_scalar_pext;
    #endif
    ChunkedCheckedScratch<L> scratch_scalar_bf;
    #if defined(FISK_HAS_SSE2)
    ChunkedCheckedScratch<L> scratch_simd_bf_sse2;
    #endif
    #if defined(FISK_HAS_AVX2)
    ChunkedCheckedScratch<L> scratch_simd_bf_avx2;
    #endif
    #if defined(FISK_HAS_AVX512)
    ChunkedCheckedScratch<L> scratch_simd_bf_avx512;
    #endif
    #if defined(FISK_HAS_NEON)
    ChunkedCheckedScratch<L> scratch_simd_bf_neon;
    #endif
    ChunkedCheckedScratch<L> scratch_simd_bf_scalar;
    #if defined(FISK_HAS_SSE2)
    ChunkedCheckedScratch<L> scratch_simd_bf_sse2_vectors;
    #endif
    #if defined(FISK_HAS_AVX2)
    ChunkedCheckedScratch<L> scratch_simd_bf_avx2_vectors;
    #endif
    #if defined(FISK_HAS_AVX512)
    ChunkedCheckedScratch<L> scratch_simd_bf_avx512_vectors;
    #endif
    #if defined(FISK_HAS_NEON)
    ChunkedCheckedScratch<L> scratch_simd_bf_neon_vectors;
    #endif
    ChunkedCheckedScratch<L> scratch_simd_bf_scalar_vectors;

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
        #if defined(FISK_HAS_BMI2)
        bench(
            "chunked_scalar_pext",
            [&](std::string const& seq) {
                return run_var_chunked_scalar_pext(
                    seq, bit_ext_mask, scratch_scalar_pext, sink_buffer
                );
            }
        ),
        #endif
        bench(
            "chunked_scalar_butterfly",
            [&](std::string const& seq) {
                return run_var_chunked_scalar_butterfly(
                    seq, bit_ext_butterfly_mask, scratch_scalar_bf, sink_buffer
                );
            }
        ),
        #if defined(FISK_HAS_SSE2)
        bench(
            "chunked_simd_butterfly_sse2_vectors",
            [&](std::string const& seq) {
                return run_var_chunked_simd_butterfly_sse2_vectors(
                    seq, simd_bf_sse2_kernel, scratch_simd_bf_sse2_vectors, sink_buffer
                );
            }
        ),
        bench(
            "chunked_simd_butterfly_sse2_scalars",
            [&](std::string const& seq) {
                return run_var_chunked_simd_butterfly_sse2_scalars(
                    seq, simd_bf_sse2_kernel, scratch_simd_bf_sse2, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_AVX2)
        bench(
            "chunked_simd_butterfly_avx2_vectors",
            [&](std::string const& seq) {
                return run_var_chunked_simd_butterfly_avx2_vectors(
                    seq, simd_bf_avx2_kernel, scratch_simd_bf_avx2_vectors, sink_buffer
                );
            }
        ),
        bench(
            "chunked_simd_butterfly_avx2_scalars",
            [&](std::string const& seq) {
                return run_var_chunked_simd_butterfly_avx2_scalars(
                    seq, simd_bf_avx2_kernel, scratch_simd_bf_avx2, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_AVX512)
        bench(
            "chunked_simd_butterfly_avx512_vectors",
            [&](std::string const& seq) {
                return run_var_chunked_simd_butterfly_avx512_vectors(
                    seq, simd_bf_avx512_kernel, scratch_simd_bf_avx512_vectors, sink_buffer
                );
            }
        ),
        bench(
            "chunked_simd_butterfly_avx512_scalars",
            [&](std::string const& seq) {
                return run_var_chunked_simd_butterfly_avx512_scalars(
                    seq, simd_bf_avx512_kernel, scratch_simd_bf_avx512, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_NEON)
        bench(
            "chunked_simd_butterfly_neon_vectors",
            [&](std::string const& seq) {
                return run_var_chunked_simd_butterfly_neon_vectors(
                    seq, simd_bf_neon_kernel, scratch_simd_bf_neon_vectors, sink_buffer
                );
            }
        ),
        bench(
            "chunked_simd_butterfly_neon_scalars",
            [&](std::string const& seq) {
                return run_var_chunked_simd_butterfly_neon_scalars(
                    seq, simd_bf_neon_kernel, scratch_simd_bf_neon, sink_buffer
                );
            }
        ),
        #endif
        bench(
            "chunked_simd_butterfly_scalar_vectors",
            [&](std::string const& seq) {
                return run_var_chunked_simd_butterfly_scalar_vectors(
                    seq, simd_bf_scalar_kernel, scratch_simd_bf_scalar_vectors, sink_buffer
                );
            }
        ),
        bench(
            "chunked_simd_butterfly_scalar_scalars",
            [&](std::string const& seq) {
                return run_var_chunked_simd_butterfly_scalar_scalars(
                    seq, simd_bf_scalar_kernel, scratch_simd_bf_scalar, sink_buffer
                );
            }
        ),
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
                    seq, bit_ext_butterfly_mask, sink_buffer
                );
            }
        ),

        // simd kernels
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
            "rolling_simd_butterfly_sse2_scalars",
            [&](std::string const& seq) {
                return run_var_rolling_simd_butterfly_sse2_scalars(
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
            "rolling_simd_butterfly_avx2_vectors",
            [&](std::string const& seq) {
                return run_var_rolling_simd_butterfly_avx2_vectors(
                    seq, simd_bf_avx2_kernel, sink_buffer
                );
            }
        ),
        bench(
            "rolling_simd_butterfly_avx2_scalars",
            [&](std::string const& seq) {
                return run_var_rolling_simd_butterfly_avx2_scalars(
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
            "rolling_simd_butterfly_avx512_vectors",
            [&](std::string const& seq) {
                return run_var_rolling_simd_butterfly_avx512_vectors(
                    seq, simd_bf_avx512_kernel, sink_buffer
                );
            }
        ),
        bench(
            "rolling_simd_butterfly_avx512_scalars",
            [&](std::string const& seq) {
                return run_var_rolling_simd_butterfly_avx512_scalars(
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
            "rolling_simd_butterfly_neon_vectors",
            [&](std::string const& seq) {
                return run_var_rolling_simd_butterfly_neon_vectors(
                    seq, simd_bf_neon_kernel, sink_buffer
                );
            }
        ),
        bench(
            "rolling_simd_butterfly_neon_scalars",
            [&](std::string const& seq) {
                return run_var_rolling_simd_butterfly_neon_scalars(
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
            "rolling_simd_pext_vectors",
            [&](std::string const& seq) {
                return run_var_rolling_simd_pext_vectors(seq, simd_pext_kernel, sink_buffer);
            }
        ),
        bench(
            "rolling_simd_pext_scalars",
            [&](std::string const& seq) {
                return run_var_rolling_simd_pext_scalars(seq, simd_pext_kernel, sink_buffer);
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
            "rolling_simd_butterfly_scalar_scalars",
            [&](std::string const& seq) {
                return run_var_rolling_simd_butterfly_scalar_scalars(
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
}

void bench_spaced_kmer_extract_ascii_checked_single(
    std::vector<std::string> const& sequences,
    std::vector<std::string> const& masks,
    std::ostream& csv_os
) {
    // User output
    std::string const suite_title = "spaced_kmer_extract_ascii_checked_single";
    std::cout << "\n=== spaced k-mer extract single mask ===\n";

    write_csv_header(csv_os);

    // The buffer of the Write sink, see sink.hpp; Sum and Barrier ignore it.
    auto sink_buffer = make_sink_buffer();

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
