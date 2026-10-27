#include <algorithm>
#include <bit>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "spaced_kmer_extract_ascii_unchecked_single/bench.hpp"
#include "microbench.hpp"
#include "utils.hpp"

using namespace fisk;

// Run the benchmarks of all variants for the masks in Layout `L`, in their own suite.run(), as
// Microbench::run() cross-validates the sinks of all benches within one call, and the Layouts emit
// different values.
template <Layout L>
static void bench_spaced_kmer_extract_ascii_unchecked_single_layout_(
    Microbench<std::string>& suite,
    std::vector<std::string> const& sequences,
    std::string const& mask,
    std::vector<std::uint64_t>& sink_buffer,
    std::ostream& csv_os,
    std::string const& suite_title,
    std::string const& case_label
) {
    // The Layout goes into the case, next to the mask, as for the other suites.
    std::string const layout_case
        = std::string("layout=") + ((L == Layout::kMSB) ? "msb" : "lsb") + ";" + case_label;

    // Prepare masks and kernels for all variants as needed
    auto const bit_ext_mask = make_spaced_kmer_mask<L>(mask);
    auto const bit_ext_butterfly_mask = make_spaced_kmer_butterfly_mask(bit_ext_mask);
    auto const simd_bf_scalar_kernel =
        make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdScalar>>(bit_ext_mask);
    #if defined(FISK_HAS_SSE2)
    auto const simd_bf_sse2_kernel =
        make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdSSE2>>(bit_ext_mask);
    #endif
    #if defined(FISK_HAS_AVX2)
    auto const simd_bf_avx2_kernel =
        make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdAVX2>>(bit_ext_mask);
    #endif
    #if defined(FISK_HAS_AVX512)
    auto const simd_bf_avx512_kernel =
        make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdAVX512>>(bit_ext_mask);
    #endif
    #if defined(FISK_HAS_NEON)
    auto const simd_bf_neon_kernel =
        make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdNEON>>(bit_ext_mask);
    #endif

    // Scratch of the variants, one each.
    #if defined(FISK_HAS_BMI2)
    ChunkedScratch<L> scratch_scalar_pext;
    #endif
    ChunkedScratch<L> scratch_scalar_bf;
    #if defined(FISK_HAS_SSE2)
    ChunkedScratch<L> scratch_simd_bf_sse2;
    #endif
    #if defined(FISK_HAS_AVX2)
    ChunkedScratch<L> scratch_simd_bf_avx2;
    #endif
    #if defined(FISK_HAS_AVX512)
    ChunkedScratch<L> scratch_simd_bf_avx512;
    #endif
    #if defined(FISK_HAS_NEON)
    ChunkedScratch<L> scratch_simd_bf_neon;
    #endif
    ChunkedScratch<L> scratch_simd_bf_scalar;
    #if defined(FISK_HAS_SSE2)
    ChunkedScratch<L> scratch_simd_bf_sse2_vectors;
    #endif
    #if defined(FISK_HAS_AVX2)
    ChunkedScratch<L> scratch_simd_bf_avx2_vectors;
    #endif
    #if defined(FISK_HAS_AVX512)
    ChunkedScratch<L> scratch_simd_bf_avx512_vectors;
    #endif
    #if defined(FISK_HAS_NEON)
    ChunkedScratch<L> scratch_simd_bf_neon_vectors;
    #endif
    ChunkedScratch<L> scratch_simd_bf_scalar_vectors;
    ChunkedScratch<L> scratch_pack_bf;

    auto results = suite.run(
        sequences,
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
            "chunked_simd_butterfly_neon_scalars",
            [&](std::string const& seq) {
                return run_var_chunked_simd_butterfly_neon_scalars(
                    seq, simd_bf_neon_kernel, scratch_simd_bf_neon, sink_buffer
                );
            }
        ),
        #endif
        bench(
            "chunked_simd_butterfly_scalar_scalars",
            [&](std::string const& seq) {
                return run_var_chunked_simd_butterfly_scalar_scalars(
                    seq, simd_bf_scalar_kernel, scratch_simd_bf_scalar, sink_buffer
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
            "pack_then_bytewise_scalar_butterfly",
            [&](std::string const& seq) {
                return run_var_pack_then_bytewise_scalar_butterfly(
                    seq, bit_ext_butterfly_mask, scratch_pack_bf, sink_buffer
                );
            }
        )
    );
    write_csv_rows(csv_os, suite_title, layout_case, results, kSinkName);
}

void bench_spaced_kmer_extract_ascii_unchecked_single(
    std::vector<std::string> const& sequences,
    std::vector<std::string> const& masks,
    std::ostream& csv_os
) {
    std::size_t const rounds = 1;
    std::size_t const repeats = 8;

    // User output
    std::string const suite_title = "spaced_kmer_extract_ascii_unchecked_single";
    std::cout << "\n=== spaced k-mer extract from ASCII, unchecked, single ===\n";
    std::cout << "rounds=" << rounds << ", repeats=" << repeats << "\n";

    write_csv_header(csv_os);

    // Backing storage for the Write sink (see sink.hpp), as in the checked suite.
    std::size_t max_seq_len = 0;
    for (auto const& seq : sequences) {
        max_seq_len = std::max(max_seq_len, seq.size());
    }
    std::vector<std::uint64_t> sink_buffer(std::bit_ceil(max_seq_len + 8), 0);

    for (std::size_t m = 0; m < masks.size(); ++m) {
        if (stdout_is_terminal()) {
            std::cout << "\rmask ";
            std::cout << std::setw(2) << (m + 1) << " / " << masks.size() << "\n";
        }
        auto const k = masks[m].size();

        Microbench<std::string> suite(suite_title);
        suite
            .rounds(rounds)
            .repeats(repeats)
            .units_fn([k](std::string const& seq) {
                // 1 unit per original (unspaced) k-mer
                return static_cast<double>(seq.size() - k + 1);
            });

        std::string const case_label = "mask=" + std::to_string(m);
        bench_spaced_kmer_extract_ascii_unchecked_single_layout_<Layout::kMSB>(
            suite, sequences, masks[m], sink_buffer, csv_os, suite_title, case_label
        );
        bench_spaced_kmer_extract_ascii_unchecked_single_layout_<Layout::kLSB>(
            suite, sequences, masks[m], sink_buffer, csv_os, suite_title, case_label
        );
    }
    if (stdout_is_terminal()) {
        std::cout << "\n";
    }
}
