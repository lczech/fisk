#include <algorithm>
#include <bit>
#include <deque>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "kmer_spaced_packed_checked_multi/bench.hpp"
#include "microbench.hpp"
#include "utils.hpp"

using namespace fisk;

// Run the benchmarks of all techniques for the masks in Layout `L`, on the sequences packed in
// that Layout, in their own suite.run(), as Microbench::run() cross-validates the sinks of all
// benches within one call, and the Layouts emit different values.
template <Layout L>
static void bench_kmer_spaced_packed_checked_multi_layout_(
    std::vector<PackedSequenceWithValidity<Encoding::kACGT, L>> const& sequences,
    std::vector<std::string> const& masks,
    std::vector<std::uint64_t>& sink_buffer,
    std::ostream& csv_os,
    std::string const& suite_title,
    std::string const& case_label,
    int rounds,
    int repeats
) {
    using Packed = PackedSequenceWithValidity<Encoding::kACGT, L>;
    std::string const prefix = (L == Layout::kMSB) ? "msb" : "lsb";

    // Benchmark names are string views, so their strings need to outlive the run.
    std::deque<std::string> names;
    auto const name = [&](std::string_view technique) -> std::string_view {
        return names.emplace_back(prefix + "_" + std::string(technique));
    };

    // Prepare a benchmark with repititions
    std::size_t const k = masks[0].size();
    double const mask_count = static_cast<double>(masks.size());
    Microbench<Packed> suite(suite_title);
    suite
        .rounds(rounds)
        .repeats(repeats)
        .units_fn([k, mask_count](Packed const& seq) {
            // 1 unit per original (unspaced) k-mer and per mask
            return static_cast<double>(seq.length() - k + 1) * mask_count;
        });

    // Prepare masks for all implementations as needed
    auto const bit_ext_masks = make_spaced_kmer_masks<L>(masks);
    auto const bit_ext_block_masks = make_spaced_kmer_block_tables(bit_ext_masks);
    auto const bit_ext_butterfly_tables = make_spaced_kmer_butterfly_tables(bit_ext_masks);

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
        make_spaced_kmer_kernels<BitExtractKernelPEXT>(bit_ext_masks)
    );
    #endif

    // Each run_() is a suite.run() of its own, as Microbench::run() cross-validates the sinks of
    // all benches within one call. The functions that hand out single valid spaced k-mers all sum
    // the same ones. The vector variants sum all lanes, valid or not, see bench.hpp, and the lanes
    // past the last spaced k-mer depend on the arch, so that each arch has its own run.
    auto const run_ = [&](auto const&... benches) {
        auto results = suite.run(sequences, benches...);
        write_csv_rows(csv_os, suite_title, case_label, results, kSinkName);
    };
    run_(
        #if defined(FISK_HAS_BMI2)
        bench(
            name("pext"),
            [&](Packed const& seq) {
                return run_var_pext(seq, bit_ext_masks, sink_buffer);
            }
        ),
        #endif
        bench(
            name("bitloop"),
            [&](Packed const& seq) {
                return run_var_bitloop(seq, bit_ext_masks, sink_buffer);
            }
        ),
        bench(
            name("byte_table"),
            [&](Packed const& seq) {
                return run_var_byte_table(seq, bit_ext_masks, sink_buffer);
            }
        ),
        bench(
            name("block_table"),
            [&](Packed const& seq) {
                return run_var_block_table(seq, bit_ext_block_masks, sink_buffer);
            }
        ),
        bench(
            name("block_table_unrolled2"),
            [&](Packed const& seq) {
                return run_var_block_table_unrolled2(seq, bit_ext_block_masks, sink_buffer);
            }
        ),
        bench(
            name("block_table_unrolled4"),
            [&](Packed const& seq) {
                return run_var_block_table_unrolled4(seq, bit_ext_block_masks, sink_buffer);
            }
        ),
        bench(
            name("block_table_unrolled8"),
            [&](Packed const& seq) {
                return run_var_block_table_unrolled8(seq, bit_ext_block_masks, sink_buffer);
            }
        ),
        bench(
            name("butterfly_table"),
            [&](Packed const& seq) {
                return run_var_butterfly_table(seq, bit_ext_butterfly_tables, sink_buffer);
            }
        ),
        #if defined(FISK_HAS_SSE2)
        bench(
            name("simd_butterfly_table_sse2_scalars"),
            [&](Packed const& seq) {
                return run_var_simd_butterfly_table_sse2_scalars(
                    seq, simd_bf_sse2_kernel, sink_buffer
                );
            }
        ),
        bench(
            name("simd_block_table_sse2_scalars"),
            [&](Packed const& seq) {
                return run_var_simd_block_table_sse2_scalars(
                    seq, simd_bt_sse2_kernel, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_AVX2)
        bench(
            name("simd_butterfly_table_avx2_scalars"),
            [&](Packed const& seq) {
                return run_var_simd_butterfly_table_avx2_scalars(
                    seq, simd_bf_avx2_kernel, sink_buffer
                );
            }
        ),
        bench(
            name("simd_block_table_avx2_scalars"),
            [&](Packed const& seq) {
                return run_var_simd_block_table_avx2_scalars(
                    seq, simd_bt_avx2_kernel, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_AVX512)
        bench(
            name("simd_butterfly_table_avx512_scalars"),
            [&](Packed const& seq) {
                return run_var_simd_butterfly_table_avx512_scalars(
                    seq, simd_bf_avx512_kernel, sink_buffer
                );
            }
        ),
        bench(
            name("simd_block_table_avx512_scalars"),
            [&](Packed const& seq) {
                return run_var_simd_block_table_avx512_scalars(
                    seq, simd_bt_avx512_kernel, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_NEON)
        bench(
            name("simd_butterfly_table_neon_scalars"),
            [&](Packed const& seq) {
                return run_var_simd_butterfly_table_neon_scalars(
                    seq, simd_bf_neon_kernel, sink_buffer
                );
            }
        ),
        bench(
            name("simd_block_table_neon_scalars"),
            [&](Packed const& seq) {
                return run_var_simd_block_table_neon_scalars(
                    seq, simd_bt_neon_kernel, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_BMI2)
        bench(
            name("simd_pext_scalars"),
            [&](Packed const& seq) {
                return run_var_simd_pext_scalars(seq, simd_pext_kernel, sink_buffer);
            }
        ),
        #endif
        bench(
            name("simd_butterfly_table_scalar_scalars"),
            [&](Packed const& seq) {
                return run_var_simd_butterfly_table_scalar_scalars(
                    seq, simd_bf_scalar_kernel, sink_buffer
                );
            }
        ),
        bench(
            name("simd_block_table_scalar_scalars"),
            [&](Packed const& seq) {
                return run_var_simd_block_table_scalar_scalars(
                    seq, simd_bt_scalar_kernel, sink_buffer
                );
            }
        )
    );
    #if defined(FISK_HAS_SSE2)
    run_(
        bench(
            name("simd_butterfly_table_sse2_vectors"),
            [&](Packed const& seq) {
                return run_var_simd_butterfly_table_sse2_vectors(
                    seq, simd_bf_sse2_kernel, sink_buffer
                );
            }
        ),
        bench(
            name("simd_block_table_sse2_vectors"),
            [&](Packed const& seq) {
                return run_var_simd_block_table_sse2_vectors(
                    seq, simd_bt_sse2_kernel, sink_buffer
                );
            }
        )
    );
    #endif
    #if defined(FISK_HAS_AVX2)
    run_(
        bench(
            name("simd_butterfly_table_avx2_vectors"),
            [&](Packed const& seq) {
                return run_var_simd_butterfly_table_avx2_vectors(
                    seq, simd_bf_avx2_kernel, sink_buffer
                );
            }
        ),
        bench(
            name("simd_block_table_avx2_vectors"),
            [&](Packed const& seq) {
                return run_var_simd_block_table_avx2_vectors(
                    seq, simd_bt_avx2_kernel, sink_buffer
                );
            }
        )
    );
    #endif
    #if defined(FISK_HAS_AVX512)
    run_(
        bench(
            name("simd_butterfly_table_avx512_vectors"),
            [&](Packed const& seq) {
                return run_var_simd_butterfly_table_avx512_vectors(
                    seq, simd_bf_avx512_kernel, sink_buffer
                );
            }
        ),
        bench(
            name("simd_block_table_avx512_vectors"),
            [&](Packed const& seq) {
                return run_var_simd_block_table_avx512_vectors(
                    seq, simd_bt_avx512_kernel, sink_buffer
                );
            }
        )
    );
    #endif
    #if defined(FISK_HAS_NEON)
    run_(
        bench(
            name("simd_butterfly_table_neon_vectors"),
            [&](Packed const& seq) {
                return run_var_simd_butterfly_table_neon_vectors(
                    seq, simd_bf_neon_kernel, sink_buffer
                );
            }
        ),
        bench(
            name("simd_block_table_neon_vectors"),
            [&](Packed const& seq) {
                return run_var_simd_block_table_neon_vectors(
                    seq, simd_bt_neon_kernel, sink_buffer
                );
            }
        )
    );
    #endif
    run_(
        #if defined(FISK_HAS_BMI2)
        bench(
            name("simd_pext_vectors"),
            [&](Packed const& seq) {
                return run_var_simd_pext_vectors(seq, simd_pext_kernel, sink_buffer);
            }
        ),
        #endif
        bench(
            name("simd_butterfly_table_scalar_vectors"),
            [&](Packed const& seq) {
                return run_var_simd_butterfly_table_scalar_vectors(
                    seq, simd_bf_scalar_kernel, sink_buffer
                );
            }
        ),
        bench(
            name("simd_block_table_scalar_vectors"),
            [&](Packed const& seq) {
                return run_var_simd_block_table_scalar_vectors(
                    seq, simd_bt_scalar_kernel, sink_buffer
                );
            }
        )
    );
}

void bench_kmer_spaced_packed_checked_multi(
    std::vector<std::string> const& sequences,
    std::vector<std::vector<std::string>> const& multi_masks,
    std::ostream& csv_os
) {
    // Check masks for consistent value of k
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

    int const rounds = 1;
    int const repeats = 4;

    // User output
    std::string const suite_title = "kmer_spaced_packed_checked_multi";
    std::cout << "\n=== spaced k-mer extract from packed sequences multi masks (checked) ===\n";
    std::cout << "rounds=" << rounds << ", repeats=" << repeats << "\n";

    write_csv_header(csv_os);

    // Pack the sequences once per Layout, outside of the timed region.
    std::vector<CheckedMsb> packed_msb;
    std::vector<CheckedLsb> packed_lsb;
    packed_msb.reserve(sequences.size());
    packed_lsb.reserve(sequences.size());
    for (auto const& seq : sequences) {
        packed_msb.push_back(
            pack_sequence_with_validity(seq, WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>{})
        );
        packed_lsb.push_back(
            pack_sequence_with_validity(seq, WordEncoderButterfly<Encoding::kACGT, Layout::kLSB>{})
        );
    }

    // Backing storage for the Write sink (see sink.hpp), sized to the longest sequence actually
    // benchmarked, times the largest mask set, rounded up to a power of two so that no call here
    // ever wraps it, as in bench_kmer_spaced_multi(). The `_vectors` variants also write the lanes
    // past the last spaced k-mer of each mask, at most 8.
    std::size_t max_seq_len = 0;
    for (auto const& seq : sequences) {
        max_seq_len = std::max(max_seq_len, seq.size());
    }
    std::size_t max_masks = 1;
    for (auto const& masks : multi_masks) {
        max_masks = std::max(max_masks, masks.size());
    }
    std::vector<std::uint64_t> sink_buffer(
        std::bit_ceil((max_seq_len + 8) * max_masks), 0
    );

    // Run a benchmark for each mask set, in both Layouts
    for (std::size_t m = 0; m < multi_masks.size(); ++m) {
        if (stdout_is_terminal()) {
            std::cout << "\rmask set ";
            std::cout << std::setw(2) << (m + 1) << " / " << multi_masks.size() << "\n";
        }
        std::string const case_label = "mask_set=" + std::to_string(m);
        bench_kmer_spaced_packed_checked_multi_layout_<Layout::kMSB>(
            packed_msb, multi_masks[m], sink_buffer, csv_os, suite_title, case_label,
            rounds, repeats
        );
        bench_kmer_spaced_packed_checked_multi_layout_<Layout::kLSB>(
            packed_lsb, multi_masks[m], sink_buffer, csv_os, suite_title, case_label,
            rounds, repeats
        );
    }
    if (stdout_is_terminal()) {
        std::cout << "\n";
    }
}
