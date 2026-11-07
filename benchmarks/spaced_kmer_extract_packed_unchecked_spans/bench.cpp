#include <algorithm>
#include <bit>
#include <iostream>
#include <string>
#include <vector>

#include "fisk/bit_extract/simd.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"
#include "spaced_kmer_extract_packed_unchecked_spans/bench.hpp"
#include "microbench.hpp"
#include "utils.hpp"

using namespace fisk;

// Run the benchmarks of all techniques for the mask of `span` in Layout `L`, on the sequences
// packed in that Layout, in their own suite.run(), as Microbench::run() cross-validates the sinks
// of all benches within one call, and the Layouts emit different values.
template <Layout L>
static void bench_spaced_kmer_extract_packed_unchecked_spans_layout_(
    std::vector<PackedSequence<Encoding::kACGT, L>> const& sequences,
    std::size_t span,
    std::vector<std::uint64_t>& sink_buffer,
    std::ostream& csv_os,
    std::string const& suite_title
) {
    using Packed = PackedSequence<Encoding::kACGT, L>;
    std::string const layout_case = std::string("layout=") + ((L == Layout::kMSB) ? "msb" : "lsb")
        + ";span=" + std::to_string(span);

    Microbench<Packed> suite(suite_title);
    suite
        .units_fn([span](Packed const& seq) {
            // 1 unit per position at which a span starts
            return seq.length() >= span ? static_cast<double>(seq.length() - span + 1) : 0.0;
        });

    auto const mask = make_spaced_kmer_mask<L, kSpanSweepMaxSpan>(spaced_mask_for_span(span));
    auto const butterfly_mask = make_spaced_kmer_butterfly_mask(mask);
    auto const bf_scalar_kernel
        = make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdScalar>>(mask);
    #if defined(FISK_HAS_SSE2)
    auto const bf_sse2_kernel
        = make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdSSE2>>(mask);
    #endif
    #if defined(FISK_HAS_AVX2)
    auto const bf_avx2_kernel
        = make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdAVX2>>(mask);
    #endif
    #if defined(FISK_HAS_AVX512)
    auto const bf_avx512_kernel
        = make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdAVX512>>(mask);
    #endif
    #if defined(FISK_HAS_NEON)
    auto const bf_neon_kernel
        = make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdNEON>>(mask);
    #endif
    #if defined(FISK_HAS_BMI2)
    auto const pext_kernel
        = make_spaced_kmer_kernel<BitExtractKernelPext>(mask);
    #endif
    auto results = suite.run(
        sequences,
        #if defined(FISK_HAS_BMI2)
        bench(
            "bytewise_scalar_pext",
            [&](Packed const& seq) {
                return run_var_bytewise_scalar_pext(seq, mask, sink_buffer);
            }
        ),
        #endif
        bench(
            "bytewise_simd_butterfly_scalar_vectors",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_butterfly_scalar_vectors(
                    seq, bf_scalar_kernel, sink_buffer
                );
            }
        ),
        #if defined(FISK_HAS_SSE2)
        bench(
            "bytewise_simd_butterfly_sse2_vectors",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_butterfly_sse2_vectors(
                    seq, bf_sse2_kernel, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_AVX2)
        bench(
            "bytewise_simd_butterfly_avx2_vectors",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_butterfly_avx2_vectors(
                    seq, bf_avx2_kernel, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_AVX512)
        bench(
            "bytewise_simd_butterfly_avx512_vectors",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_butterfly_avx512_vectors(
                    seq, bf_avx512_kernel, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_NEON)
        bench(
            "bytewise_simd_butterfly_neon_vectors",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_butterfly_neon_vectors(
                    seq, bf_neon_kernel, sink_buffer
                );
            }
        ),
        #endif
        #if defined(FISK_HAS_BMI2)
        bench(
            "bytewise_simd_pext_vectors",
            [&](Packed const& seq) {
                return run_var_bytewise_simd_pext_vectors(
                    seq, pext_kernel, sink_buffer
                );
            }
        ),
        #endif
        bench(
            "bytewise_scalar_butterfly",
            [&](Packed const& seq) {
                return run_var_bytewise_scalar_butterfly(seq, butterfly_mask, sink_buffer);
            }
        )
    );
    write_csv_rows(csv_os, suite_title, layout_case, results, kSinkName);
}

void bench_spaced_kmer_extract_packed_unchecked_spans(
    std::vector<std::string> const& sequences,
    std::ostream& csv_os
) {
    // User output
    std::string const suite_title = "spaced_kmer_extract_packed_unchecked_spans";
    std::cout << "\n=== spaced k-mer extract from packed sequences, spans 1 to "
        << kSpanSweepMaxSpan << " ===\n";

    write_csv_header(csv_os);

    // Pack the sequences once per Layout, outside of the timed region.
    std::vector<PackedMsb> packed_msb;
    std::vector<PackedLsb> packed_lsb;
    packed_msb.reserve(sequences.size());
    packed_lsb.reserve(sequences.size());
    for (auto const& seq : sequences) {
        packed_msb.push_back(
            pack_sequence_unchecked(seq, WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>{})
        );
        packed_lsb.push_back(
            pack_sequence_unchecked(seq, WordEncoderButterfly<Encoding::kACGT, Layout::kLSB>{})
        );
    }

    // Backing storage for the Write sink (see sink.hpp), sized to the longest sequence, rounded up
    // to a power of two, so that no call here ever wraps it.
    std::size_t max_seq_len = 0;
    for (auto const& seq : sequences) {
        max_seq_len = std::max(max_seq_len, seq.size());
    }
    std::vector<std::uint64_t> sink_buffer(std::bit_ceil(max_seq_len + 8), 0);

    for (std::size_t span = 1; span <= kSpanSweepMaxSpan; ++span) {
        if (stdout_is_terminal()) {
            std::cout << "\rspan " << span << " / " << kSpanSweepMaxSpan << std::flush;
        }
        bench_spaced_kmer_extract_packed_unchecked_spans_layout_<Layout::kMSB>(
            packed_msb, span, sink_buffer, csv_os, suite_title
        );
        bench_spaced_kmer_extract_packed_unchecked_spans_layout_<Layout::kLSB>(
            packed_lsb, span, sink_buffer, csv_os, suite_title
        );
    }
    if (stdout_is_terminal()) {
        std::cout << "\n";
    }
}
