#include <algorithm>
#include <bit>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "spaced_kmer_extract_packed_checked_single/bench.hpp"
#include "microbench.hpp"
#include "utils.hpp"

using namespace fisk;

// Run the benchmarks of all techniques for the mask in Layout `L`, on the sequences packed in that
// Layout, in their own suite.run(), as Microbench::run() cross-validates the sinks of all benches
// within one call, and the Layouts emit different values.
template <Layout L>
static void bench_spaced_kmer_extract_packed_checked_single_layout_(
    std::vector<PackedSequenceWithValidity<Encoding::kACGT, L>> const& sequences,
    std::string const& mask,
    std::vector<std::uint64_t>& sink_buffer,
    std::ostream& csv_os,
    std::string const& suite_title,
    std::string const& case_label,
    std::size_t rounds,
    std::size_t repeats
) {
    using Packed = PackedSequenceWithValidity<Encoding::kACGT, L>;
    // The Layout goes into the case, next to the mask, as for the other suites.
    std::string const layout_case
        = std::string("layout=") + ((L == Layout::kMSB) ? "msb" : "lsb") + ";" + case_label;

    // Prepare a benchmark with repititions
    std::size_t const k = mask.size();
    Microbench<Packed> suite(suite_title);
    suite
        .rounds(rounds)
        .repeats(repeats)
        .units_fn([k](Packed const& seq) {
            // 1 unit per original (unspaced) k-mer
            return static_cast<double>(seq.length() - k + 1);
        });

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

    // Each run_() is a suite.run() of its own, as Microbench::run() cross-validates the sinks of
    // all benches within one call. The functions that hand out single valid spaced k-mers all sum
    // the same ones. The vector variants sum all lanes, valid or not, see bench.hpp, and the lanes
    // past the last spaced k-mer depend on the arch, so that each arch has its own run.
    auto const run_ = [&](auto const&... benches) {
        auto results = suite.run(sequences, benches...);
        write_csv_rows(csv_os, suite_title, layout_case, results, kSinkName);
    };
    run_(
        #if defined(FISK_HAS_BMI2)
        bench(
            "bytewise_scalar_pext",
            [&](Packed const& seq) {
                return run_var_bytewise_scalar_pext(seq, bit_ext_mask, sink_buffer);
            }
        ),
        #endif
        bench(
            "bytewise_scalar_bit_loop",
            [&](Packed const& seq) {
                return run_var_bytewise_scalar_bit_loop(seq, bit_ext_mask, sink_buffer);
            }
        ),
        bench(
            "bytewise_scalar_byte_lookup",
            [&](Packed const& seq) {
                return run_var_bytewise_scalar_byte_lookup(seq, bit_ext_mask, sink_buffer);
            }
        ),
        bench(
            "bytewise_scalar_block_shift",
            [&](Packed const& seq) {
                return run_var_bytewise_scalar_block_shift(seq, bit_ext_block_mask, sink_buffer);
            }
        ),
        bench(
            "bytewise_scalar_block_shift_unrolled2",
            [&](Packed const& seq) {
                return run_var_bytewise_scalar_block_shift_unrolled2(
                    seq, bit_ext_block_mask, sink_buffer
                );
            }
        ),
        bench(
            "bytewise_scalar_block_shift_unrolled4",
            [&](Packed const& seq) {
                return run_var_bytewise_scalar_block_shift_unrolled4(
                    seq, bit_ext_block_mask, sink_buffer
                );
            }
        ),
        bench(
            "bytewise_scalar_block_shift_unrolled8",
            [&](Packed const& seq) {
                return run_var_bytewise_scalar_block_shift_unrolled8(
                    seq, bit_ext_block_mask, sink_buffer
                );
            }
        ),
        bench(
            "bytewise_scalar_butterfly",
            [&](Packed const& seq) {
                return run_var_bytewise_scalar_butterfly(
                    seq, bit_ext_butterfly_table, sink_buffer
                );
            }
        ),
        #if defined(FISK_HAS_SSE2)
        bench(
            "bytewise_simd_butterfly_sse2_scalars",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_butterfly_sse2_scalars(
                    seq, simd_bf_sse2_kernel, sink_buffer
                );
            }
        ),
        bench(
            "bytewise_simd_block_shift_sse2_scalars",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_block_shift_sse2_scalars(
                    seq, simd_bt_sse2_kernel, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_AVX2)
        bench(
            "bytewise_simd_butterfly_avx2_scalars",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_butterfly_avx2_scalars(
                    seq, simd_bf_avx2_kernel, sink_buffer
                );
            }
        ),
        bench(
            "bytewise_simd_block_shift_avx2_scalars",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_block_shift_avx2_scalars(
                    seq, simd_bt_avx2_kernel, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_AVX512)
        bench(
            "bytewise_simd_butterfly_avx512_scalars",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_butterfly_avx512_scalars(
                    seq, simd_bf_avx512_kernel, sink_buffer
                );
            }
        ),
        bench(
            "bytewise_simd_block_shift_avx512_scalars",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_block_shift_avx512_scalars(
                    seq, simd_bt_avx512_kernel, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_NEON)
        bench(
            "bytewise_simd_butterfly_neon_scalars",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_butterfly_neon_scalars(
                    seq, simd_bf_neon_kernel, sink_buffer
                );
            }
        ),
        bench(
            "bytewise_simd_block_shift_neon_scalars",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_block_shift_neon_scalars(
                    seq, simd_bt_neon_kernel, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_BMI2)
        bench(
            "bytewise_simd_pext_scalars",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_pext_scalars(seq, simd_pext_kernel, sink_buffer);
            }
        ),
        #endif
        bench(
            "bytewise_simd_butterfly_scalar_scalars",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_butterfly_scalar_scalars(
                    seq, simd_bf_scalar_kernel, sink_buffer
                );
            }
        ),
        bench(
            "bytewise_simd_block_shift_scalar_scalars",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_block_shift_scalar_scalars(
                    seq, simd_bt_scalar_kernel, sink_buffer
                );
            }
        )
    );
    #if defined(FISK_HAS_SSE2)
    run_(
        bench(
            "bytewise_simd_butterfly_sse2_vectors",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_butterfly_sse2_vectors(
                    seq, simd_bf_sse2_kernel, sink_buffer
                );
            }
        ),
        bench(
            "bytewise_simd_block_shift_sse2_vectors",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_block_shift_sse2_vectors(
                    seq, simd_bt_sse2_kernel, sink_buffer
                );
            }
        )
    );
    #endif
    #if defined(FISK_HAS_AVX2)
    run_(
        bench(
            "bytewise_simd_butterfly_avx2_vectors",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_butterfly_avx2_vectors(
                    seq, simd_bf_avx2_kernel, sink_buffer
                );
            }
        ),
        bench(
            "bytewise_simd_block_shift_avx2_vectors",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_block_shift_avx2_vectors(
                    seq, simd_bt_avx2_kernel, sink_buffer
                );
            }
        )
    );
    #endif
    #if defined(FISK_HAS_AVX512)
    run_(
        bench(
            "bytewise_simd_butterfly_avx512_vectors",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_butterfly_avx512_vectors(
                    seq, simd_bf_avx512_kernel, sink_buffer
                );
            }
        ),
        bench(
            "bytewise_simd_block_shift_avx512_vectors",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_block_shift_avx512_vectors(
                    seq, simd_bt_avx512_kernel, sink_buffer
                );
            }
        )
    );
    #endif
    #if defined(FISK_HAS_NEON)
    run_(
        bench(
            "bytewise_simd_butterfly_neon_vectors",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_butterfly_neon_vectors(
                    seq, simd_bf_neon_kernel, sink_buffer
                );
            }
        ),
        bench(
            "bytewise_simd_block_shift_neon_vectors",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_block_shift_neon_vectors(
                    seq, simd_bt_neon_kernel, sink_buffer
                );
            }
        )
    );
    #endif
    run_(
        #if defined(FISK_HAS_BMI2)
        bench(
            "bytewise_simd_pext_vectors",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_pext_vectors(seq, simd_pext_kernel, sink_buffer);
            }
        ),
        #endif
        bench(
            "bytewise_simd_butterfly_scalar_vectors",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_butterfly_scalar_vectors(
                    seq, simd_bf_scalar_kernel, sink_buffer
                );
            }
        ),
        bench(
            "bytewise_simd_block_shift_scalar_vectors",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_block_shift_scalar_vectors(
                    seq, simd_bt_scalar_kernel, sink_buffer
                );
            }
        )
    );
}

void bench_spaced_kmer_extract_packed_checked_single(
    std::vector<std::string> const& sequences,
    std::vector<std::string> const& masks,
    std::ostream& csv_os
) {
    std::size_t const rounds = 1;
    std::size_t const repeats = 8;

    // User output
    std::string const suite_title = "spaced_kmer_extract_packed_checked_single";
    std::cout << "\n=== spaced k-mer extract from packed sequences single mask (checked) ===\n";
    std::cout << "rounds=" << rounds << ", repeats=" << repeats << "\n";

    write_csv_header(csv_os);

    // Pack the sequences once per Layout, outside of the timed region.
    std::vector<CheckedMsb> packed_msb;
    std::vector<CheckedLsb> packed_lsb;
    packed_msb.reserve(sequences.size());
    packed_lsb.reserve(sequences.size());
    for (auto const& seq : sequences) {
        packed_msb.push_back(
            pack_sequence_checked(seq, WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>{})
        );
        packed_lsb.push_back(
            pack_sequence_checked(seq, WordEncoderButterfly<Encoding::kACGT, Layout::kLSB>{})
        );
    }

    // Backing storage for the Write sink (see sink.hpp), sized to the longest sequence actually
    // benchmarked (rounded up to a power of two) so that no call here ever wraps it, as in
    // bench_spaced_kmer_extract_ascii_checked_single(). The `_vectors` variants also write the
    // lanes past the last spaced k-mer, at most 8.
    std::size_t max_seq_len = 0;
    for (auto const& seq : sequences) {
        max_seq_len = std::max(max_seq_len, seq.size());
    }
    std::vector<std::uint64_t> sink_buffer(std::bit_ceil(max_seq_len + 8), 0);

    // Run a benchmark for each mask, in both Layouts
    for (std::size_t m = 0; m < masks.size(); ++m) {
        if (stdout_is_terminal()) {
            std::cout << "\rmask ";
            std::cout << std::setw(2) << (m + 1) << " / " << masks.size() << "\n";
        }
        std::string const case_label = "mask=" + std::to_string(m);
        bench_spaced_kmer_extract_packed_checked_single_layout_<Layout::kMSB>(
            packed_msb, masks[m], sink_buffer, csv_os, suite_title, case_label, rounds, repeats
        );
        bench_spaced_kmer_extract_packed_checked_single_layout_<Layout::kLSB>(
            packed_lsb, masks[m], sink_buffer, csv_os, suite_title, case_label, rounds, repeats
        );
    }
    if (stdout_is_terminal()) {
        std::cout << "\n";
    }
}
