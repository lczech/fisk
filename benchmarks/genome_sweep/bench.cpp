#include <cstddef>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "fisk/extraction/estimate.hpp"
#include "fisk/extraction/for_each.hpp"
#include "fisk/util/cpu_runtime.hpp"
#include "genome_sweep/bench.hpp"
#include "kmer_extract_ascii_checked/bench.hpp"
#include "kmer_extract_packed_checked/bench.hpp"
#include "spaced_kmer_extract_ascii_checked_single/bench.hpp"
#include "spaced_kmer_extract_packed_checked_single/bench.hpp"
#include "chunked.hpp"
#include "microbench.hpp"
#include "utils.hpp"

using namespace fisk;

// The number of k-mer positions of a sequence of `n` nucleotides, the unit of all cases.
static double genome_sweep_units_(std::size_t n, std::size_t k)
{
    return n >= k ? static_cast<double>(n - k + 1) : 0.0;
}

void bench_genome_sweep(
    std::vector<InputSequence> const& sequences,
    std::string const& mask,
    std::ostream& csv_os
) {
    constexpr auto C = InputCheck::kChecked;
    constexpr auto E = Encoding::kACGT;
    constexpr auto L = Layout::kLSB;
    constexpr std::size_t k = kScalingK;
    std::size_t const span = mask.size();
    std::string const suite_title = "genome_sweep";
    std::cout << "\n=== genome sweep ===\n";
    write_csv_header(csv_os);

    // The buffer of the Write sink, see sink.hpp; Sum and Barrier ignore it.
    auto sink_buffer = make_sink_buffer();

    // The algorithms with PEXT only run where it is likely fast, see pext_likely_fast(), as they
    // take much longer than the others elsewhere, such as on AMD Zen 2.
    [[maybe_unused]] bool const pext_fast = pext_likely_fast();

    // -------------------------------------------------------------------------
    //     Packing
    // -------------------------------------------------------------------------

    // The packing of the packed cases below, and that of the estimate's word encoder, which the
    // chunked loops use for their chunks. The sink is the number of nucleotides, the same for both.
    {
        Microbench<InputSequence> suite(suite_title);
        suite.units_fn([](InputSequence const& seq) { return static_cast<double>(seq.size()); });
        auto results = suite.run(
            sequences,
            bench(
                "pack_sequence_checked_butterfly",
                [](InputSequence const& seq) {
                    return pack_sequence_checked(seq, WordEncoderButterfly<E, L>{}).length();
                }
            ),
            bench(
                "pack_sequence_checked_estimate",
                [](InputSequence const& seq) {
                    return pack_sequence_checked(seq, DefaultWordEncoder<E, L>{}).length();
                }
            )
        );
        write_csv_rows(csv_os, suite_title, "input=ascii;kind=pack", results, kSinkName);
    }

    // -------------------------------------------------------------------------
    //     k-mers from ASCII
    // -------------------------------------------------------------------------

    {
        // Each scratch buffer stays dedicated to one variant, as in the suites.
        CheckedAcgtLsb scratch_scalar;
        #if defined(FISK_HAS_SSE2)
        CheckedAcgtLsb scratch_sse2;
        CheckedAcgtLsb scratch_sse2_scalars;
        #endif
        #if defined(FISK_HAS_AVX2)
        CheckedAcgtLsb scratch_avx2;
        CheckedAcgtLsb scratch_avx2_scalars;
        #endif
        #if defined(FISK_HAS_AVX512)
        CheckedAcgtLsb scratch_avx512;
        CheckedAcgtLsb scratch_avx512_scalars;
        #endif
        #if defined(FISK_HAS_NEON)
        CheckedAcgtLsb scratch_neon;
        CheckedAcgtLsb scratch_neon_scalars;
        #endif

        Microbench<InputSequence> suite(suite_title);
        suite.units_fn([](InputSequence const& seq) { return genome_sweep_units_(seq.size(), k); });
        auto results = suite.run(
            sequences,
            bench(
                "for_each_kmer",
                [&](InputSequence const& seq) { return run_var_kmer_checked(seq, sink_buffer); }
            ),
            bench(
                "for_each_kmer_vectors",
                [&](InputSequence const& seq) {
                    return run_var_kmer_checked_vectors(seq, sink_buffer);
                }
            ),
            bench(
                "chunked_scalar",
                [&](InputSequence const& seq) {
                    return run_var_chunked_scalar_acgt_lsb(seq, k, scratch_scalar, sink_buffer);
                }
            ),
            #if defined(FISK_HAS_SSE2)
            bench(
                "chunked_simd_sse2",
                [&](InputSequence const& seq) {
                    return run_var_chunked_simd_sse2_acgt_lsb(seq, k, scratch_sse2, sink_buffer);
                }
            ),
            bench(
                "chunked_simd_sse2_scalars",
                [&](InputSequence const& seq) {
                    return run_var_chunked_simd_sse2_scalars_acgt_lsb(
                        seq, k, scratch_sse2_scalars, sink_buffer
                    );
                }
            ),
            bench(
                "rolling_simd_sse2",
                [&](InputSequence const& seq) {
                    return run_var_rolling_simd_sse2_acgt_lsb(seq, k, sink_buffer);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX2)
            bench(
                "chunked_simd_avx2",
                [&](InputSequence const& seq) {
                    return run_var_chunked_simd_avx2_acgt_lsb(seq, k, scratch_avx2, sink_buffer);
                }
            ),
            bench(
                "chunked_simd_avx2_scalars",
                [&](InputSequence const& seq) {
                    return run_var_chunked_simd_avx2_scalars_acgt_lsb(
                        seq, k, scratch_avx2_scalars, sink_buffer
                    );
                }
            ),
            bench(
                "rolling_simd_avx2",
                [&](InputSequence const& seq) {
                    return run_var_rolling_simd_avx2_acgt_lsb(seq, k, sink_buffer);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX512)
            bench(
                "chunked_simd_avx512",
                [&](InputSequence const& seq) {
                    return run_var_chunked_simd_avx512_acgt_lsb(
                        seq, k, scratch_avx512, sink_buffer
                    );
                }
            ),
            bench(
                "chunked_simd_avx512_scalars",
                [&](InputSequence const& seq) {
                    return run_var_chunked_simd_avx512_scalars_acgt_lsb(
                        seq, k, scratch_avx512_scalars, sink_buffer
                    );
                }
            ),
            bench(
                "rolling_simd_avx512",
                [&](InputSequence const& seq) {
                    return run_var_rolling_simd_avx512_acgt_lsb(seq, k, sink_buffer);
                }
            ),
            #endif
            #if defined(FISK_HAS_NEON)
            bench(
                "chunked_simd_neon",
                [&](InputSequence const& seq) {
                    return run_var_chunked_simd_neon_acgt_lsb(seq, k, scratch_neon, sink_buffer);
                }
            ),
            bench(
                "chunked_simd_neon_scalars",
                [&](InputSequence const& seq) {
                    return run_var_chunked_simd_neon_scalars_acgt_lsb(
                        seq, k, scratch_neon_scalars, sink_buffer
                    );
                }
            ),
            bench(
                "rolling_simd_neon",
                [&](InputSequence const& seq) {
                    return run_var_rolling_simd_neon_acgt_lsb(seq, k, sink_buffer);
                }
            ),
            #endif
            bench(
                "rolling_scalar_table",
                [&](InputSequence const& seq) {
                    return run_var_rolling_scalar_table_acgt_lsb(seq, k, sink_buffer);
                }
            )
        );
        std::string const estimate = for_each_kmer_algorithm_name<C, E, L>(std::string_view{}, k);
        write_csv_rows(
            csv_os, suite_title, "input=ascii;kind=kmers;estimate=" + estimate, results, kSinkName
        );
    }

    // -------------------------------------------------------------------------
    //     k-mers from packed input
    // -------------------------------------------------------------------------

    // Packed once, for the k-mers and the spaced k-mers.
    std::vector<GenomeSweepPacked> packed;
    packed.reserve(sequences.size());
    for (auto const& seq : sequences) {
        packed.push_back(pack_sequence_checked(seq, WordEncoderButterfly<E, L>{}));
    }

    {
        Microbench<GenomeSweepPacked> suite(suite_title);
        suite.units_fn([](GenomeSweepPacked const& seq) {
            return genome_sweep_units_(seq.length(), k);
        });
        auto results = suite.run(
            packed,
            bench(
                "for_each_kmer",
                [&](GenomeSweepPacked const& seq) { return run_var_kmer_packed(seq, sink_buffer); }
            ),
            bench(
                "for_each_kmer_vectors",
                [&](GenomeSweepPacked const& seq) {
                    return run_var_kmer_packed_vectors(seq, sink_buffer);
                }
            ),
            bench(
                "bytewise_scalar",
                [&](GenomeSweepPacked const& seq) {
                    return run_var_checked_bytewise_scalar_lsb(seq, k, sink_buffer);
                }
            ),
            #if defined(FISK_HAS_SSE2)
            bench(
                "bytewise_simd_sse2",
                [&](GenomeSweepPacked const& seq) {
                    return run_var_checked_bytewise_simd_sse2_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_sse2_scalars",
                [&](GenomeSweepPacked const& seq) {
                    return run_var_checked_bytewise_simd_sse2_scalars_lsb(seq, k, sink_buffer);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX2)
            bench(
                "bytewise_simd_avx2",
                [&](GenomeSweepPacked const& seq) {
                    return run_var_checked_bytewise_simd_avx2_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_avx2_scalars",
                [&](GenomeSweepPacked const& seq) {
                    return run_var_checked_bytewise_simd_avx2_scalars_lsb(seq, k, sink_buffer);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX512)
            bench(
                "bytewise_simd_avx512",
                [&](GenomeSweepPacked const& seq) {
                    return run_var_checked_bytewise_simd_avx512_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_avx512_scalars",
                [&](GenomeSweepPacked const& seq) {
                    return run_var_checked_bytewise_simd_avx512_scalars_lsb(seq, k, sink_buffer);
                }
            ),
            #endif
            #if defined(FISK_HAS_NEON)
            bench(
                "bytewise_simd_neon",
                [&](GenomeSweepPacked const& seq) {
                    return run_var_checked_bytewise_simd_neon_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_neon_scalars",
                [&](GenomeSweepPacked const& seq) {
                    return run_var_checked_bytewise_simd_neon_scalars_lsb(seq, k, sink_buffer);
                }
            ),
            #endif
            bench(
                "rolling_scalar",
                [&](GenomeSweepPacked const& seq) {
                    return run_var_checked_rolling_scalar_lsb(seq, k, sink_buffer);
                }
            )
        );
        std::string const estimate = packed.empty()
            ? std::string()
            : for_each_kmer_algorithm_name(packed.front(), k);
        write_csv_rows(
            csv_os, suite_title, "input=packed;kind=kmers;estimate=" + estimate, results, kSinkName
        );
    }

    // -------------------------------------------------------------------------
    //     Spaced k-mers, masks
    // -------------------------------------------------------------------------

    // The forms of the mask for the estimate, and for the variants, as in the suites.
    ScalingMaskForms const mask_forms(std::vector<std::string>{ mask });
    auto const bit_ext_mask = make_spaced_kmer_mask<L>(mask);
    auto const bit_ext_butterfly_mask = make_spaced_kmer_butterfly_mask(bit_ext_mask);
    #if defined(FISK_HAS_SSE2)
    auto const bf_sse2_kernel
        = make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdSSE2>>(bit_ext_mask);
    #endif
    #if defined(FISK_HAS_AVX2)
    auto const bf_avx2_kernel
        = make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdAVX2>>(bit_ext_mask);
    #endif
    #if defined(FISK_HAS_AVX512)
    auto const bf_avx512_kernel
        = make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdAVX512>>(bit_ext_mask);
    #endif
    #if defined(FISK_HAS_NEON)
    auto const bf_neon_kernel
        = make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdNEON>>(bit_ext_mask);
    #endif
    #if defined(FISK_HAS_BMI2)
    auto const pext_kernel = make_spaced_kmer_kernel<BitExtractKernelPext>(bit_ext_mask);
    #endif

    // -------------------------------------------------------------------------
    //     Spaced k-mers from ASCII
    // -------------------------------------------------------------------------

    {
        #if defined(FISK_HAS_BMI2)
        ChunkedCheckedScratch<L> scratch_pext;
        #endif
        ChunkedCheckedScratch<L> scratch_butterfly;
        #if defined(FISK_HAS_SSE2)
        ChunkedCheckedScratch<L> scratch_sse2;
        ChunkedCheckedScratch<L> scratch_sse2_scalars;
        #endif
        #if defined(FISK_HAS_AVX2)
        ChunkedCheckedScratch<L> scratch_avx2;
        ChunkedCheckedScratch<L> scratch_avx2_scalars;
        #endif
        #if defined(FISK_HAS_AVX512)
        ChunkedCheckedScratch<L> scratch_avx512;
        ChunkedCheckedScratch<L> scratch_avx512_scalars;
        #endif
        #if defined(FISK_HAS_NEON)
        ChunkedCheckedScratch<L> scratch_neon;
        ChunkedCheckedScratch<L> scratch_neon_scalars;
        #endif

        Microbench<InputSequence> suite(suite_title);
        suite.units_fn([span](InputSequence const& seq) {
            return genome_sweep_units_(seq.size(), span);
        });
        auto results = suite.run(
            sequences,
            bench(
                "for_each_spaced_kmer",
                [&](InputSequence const& seq) {
                    return run_var_spaced_kmer_checked(seq, mask_forms, sink_buffer);
                }
            ),
            bench(
                "for_each_spaced_kmer_vectors",
                [&](InputSequence const& seq) {
                    return run_var_spaced_kmer_checked_vectors(seq, mask_forms, sink_buffer);
                }
            ),
            #if defined(FISK_HAS_BMI2)
            bench_if(
                pext_fast, "chunked_scalar_pext",
                [&](InputSequence const& seq) {
                    return run_var_chunked_scalar_pext(
                        seq, bit_ext_mask, scratch_pext, sink_buffer
                    );
                }
            ),
            bench_if(
                pext_fast, "rolling_scalar_pext",
                [&](InputSequence const& seq) {
                    return run_var_rolling_scalar_pext(seq, bit_ext_mask, sink_buffer);
                }
            ),
            bench_if(
                pext_fast, "rolling_simd_pext_vectors",
                [&](InputSequence const& seq) {
                    return run_var_rolling_simd_pext_vectors(seq, pext_kernel, sink_buffer);
                }
            ),
            bench_if(
                pext_fast, "rolling_simd_pext_scalars",
                [&](InputSequence const& seq) {
                    return run_var_rolling_simd_pext_scalars(seq, pext_kernel, sink_buffer);
                }
            ),
            #endif
            bench(
                "chunked_scalar_butterfly",
                [&](InputSequence const& seq) {
                    return run_var_chunked_scalar_butterfly(
                        seq, bit_ext_butterfly_mask, scratch_butterfly, sink_buffer
                    );
                }
            ),
            #if defined(FISK_HAS_SSE2)
            bench(
                "chunked_simd_butterfly_sse2_vectors",
                [&](InputSequence const& seq) {
                    return run_var_chunked_simd_butterfly_sse2_vectors(
                        seq, bf_sse2_kernel, scratch_sse2, sink_buffer
                    );
                }
            ),
            bench(
                "chunked_simd_butterfly_sse2_scalars",
                [&](InputSequence const& seq) {
                    return run_var_chunked_simd_butterfly_sse2_scalars(
                        seq, bf_sse2_kernel, scratch_sse2_scalars, sink_buffer
                    );
                }
            ),
            bench(
                "rolling_simd_butterfly_sse2_vectors",
                [&](InputSequence const& seq) {
                    return run_var_rolling_simd_butterfly_sse2_vectors(
                        seq, bf_sse2_kernel, sink_buffer
                    );
                }
            ),
            bench(
                "rolling_simd_butterfly_sse2_scalars",
                [&](InputSequence const& seq) {
                    return run_var_rolling_simd_butterfly_sse2_scalars(
                        seq, bf_sse2_kernel, sink_buffer
                    );
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX2)
            bench(
                "chunked_simd_butterfly_avx2_vectors",
                [&](InputSequence const& seq) {
                    return run_var_chunked_simd_butterfly_avx2_vectors(
                        seq, bf_avx2_kernel, scratch_avx2, sink_buffer
                    );
                }
            ),
            bench(
                "chunked_simd_butterfly_avx2_scalars",
                [&](InputSequence const& seq) {
                    return run_var_chunked_simd_butterfly_avx2_scalars(
                        seq, bf_avx2_kernel, scratch_avx2_scalars, sink_buffer
                    );
                }
            ),
            bench(
                "rolling_simd_butterfly_avx2_vectors",
                [&](InputSequence const& seq) {
                    return run_var_rolling_simd_butterfly_avx2_vectors(
                        seq, bf_avx2_kernel, sink_buffer
                    );
                }
            ),
            bench(
                "rolling_simd_butterfly_avx2_scalars",
                [&](InputSequence const& seq) {
                    return run_var_rolling_simd_butterfly_avx2_scalars(
                        seq, bf_avx2_kernel, sink_buffer
                    );
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX512)
            bench(
                "chunked_simd_butterfly_avx512_vectors",
                [&](InputSequence const& seq) {
                    return run_var_chunked_simd_butterfly_avx512_vectors(
                        seq, bf_avx512_kernel, scratch_avx512, sink_buffer
                    );
                }
            ),
            bench(
                "chunked_simd_butterfly_avx512_scalars",
                [&](InputSequence const& seq) {
                    return run_var_chunked_simd_butterfly_avx512_scalars(
                        seq, bf_avx512_kernel, scratch_avx512_scalars, sink_buffer
                    );
                }
            ),
            bench(
                "rolling_simd_butterfly_avx512_vectors",
                [&](InputSequence const& seq) {
                    return run_var_rolling_simd_butterfly_avx512_vectors(
                        seq, bf_avx512_kernel, sink_buffer
                    );
                }
            ),
            bench(
                "rolling_simd_butterfly_avx512_scalars",
                [&](InputSequence const& seq) {
                    return run_var_rolling_simd_butterfly_avx512_scalars(
                        seq, bf_avx512_kernel, sink_buffer
                    );
                }
            ),
            #endif
            #if defined(FISK_HAS_NEON)
            bench(
                "chunked_simd_butterfly_neon_vectors",
                [&](InputSequence const& seq) {
                    return run_var_chunked_simd_butterfly_neon_vectors(
                        seq, bf_neon_kernel, scratch_neon, sink_buffer
                    );
                }
            ),
            bench(
                "chunked_simd_butterfly_neon_scalars",
                [&](InputSequence const& seq) {
                    return run_var_chunked_simd_butterfly_neon_scalars(
                        seq, bf_neon_kernel, scratch_neon_scalars, sink_buffer
                    );
                }
            ),
            bench(
                "rolling_simd_butterfly_neon_vectors",
                [&](InputSequence const& seq) {
                    return run_var_rolling_simd_butterfly_neon_vectors(
                        seq, bf_neon_kernel, sink_buffer
                    );
                }
            ),
            bench(
                "rolling_simd_butterfly_neon_scalars",
                [&](InputSequence const& seq) {
                    return run_var_rolling_simd_butterfly_neon_scalars(
                        seq, bf_neon_kernel, sink_buffer
                    );
                }
            ),
            #endif
            bench(
                "rolling_scalar_butterfly",
                [&](InputSequence const& seq) {
                    return run_var_rolling_scalar_butterfly(
                        seq, bit_ext_butterfly_mask, sink_buffer
                    );
                }
            )
        );
        std::string const estimate
            = for_each_spaced_kmer_algorithm_name<C, E, L>(std::string_view{}, mask_forms);
        write_csv_rows(
            csv_os, suite_title, "input=ascii;kind=spaced_kmers;estimate=" + estimate, results,
            kSinkName
        );
    }

    // -------------------------------------------------------------------------
    //     Spaced k-mers from packed input
    // -------------------------------------------------------------------------

    {
        Microbench<GenomeSweepPacked> suite(suite_title);
        suite.units_fn([span](GenomeSweepPacked const& seq) {
            return genome_sweep_units_(seq.length(), span);
        });
        auto results = suite.run(
            packed,
            bench(
                "for_each_spaced_kmer",
                [&](GenomeSweepPacked const& seq) {
                    return run_var_spaced_kmer_packed(seq, mask_forms, sink_buffer);
                }
            ),
            bench(
                "for_each_spaced_kmer_vectors",
                [&](GenomeSweepPacked const& seq) {
                    return run_var_spaced_kmer_packed_vectors(seq, mask_forms, sink_buffer);
                }
            ),
            #if defined(FISK_HAS_BMI2)
            bench_if(
                pext_fast, "bytewise_scalar_pext",
                [&](GenomeSweepPacked const& seq) {
                    return run_var_bytewise_scalar_pext(seq, bit_ext_mask, sink_buffer);
                }
            ),
            bench_if(
                pext_fast, "bytewise_simd_pext_vectors",
                [&](GenomeSweepPacked const& seq) {
                    return run_var_bytewise_simd_pext_vectors(seq, pext_kernel, sink_buffer);
                }
            ),
            bench_if(
                pext_fast, "bytewise_simd_pext_scalars",
                [&](GenomeSweepPacked const& seq) {
                    return run_var_bytewise_simd_pext_scalars(seq, pext_kernel, sink_buffer);
                }
            ),
            #endif
            #if defined(FISK_HAS_SSE2)
            bench(
                "bytewise_simd_butterfly_sse2_vectors",
                [&](GenomeSweepPacked const& seq) {
                    return run_var_bytewise_simd_butterfly_sse2_vectors(
                        seq, bf_sse2_kernel, sink_buffer
                    );
                }
            ),
            bench(
                "bytewise_simd_butterfly_sse2_scalars",
                [&](GenomeSweepPacked const& seq) {
                    return run_var_bytewise_simd_butterfly_sse2_scalars(
                        seq, bf_sse2_kernel, sink_buffer
                    );
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX2)
            bench(
                "bytewise_simd_butterfly_avx2_vectors",
                [&](GenomeSweepPacked const& seq) {
                    return run_var_bytewise_simd_butterfly_avx2_vectors(
                        seq, bf_avx2_kernel, sink_buffer
                    );
                }
            ),
            bench(
                "bytewise_simd_butterfly_avx2_scalars",
                [&](GenomeSweepPacked const& seq) {
                    return run_var_bytewise_simd_butterfly_avx2_scalars(
                        seq, bf_avx2_kernel, sink_buffer
                    );
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX512)
            bench(
                "bytewise_simd_butterfly_avx512_vectors",
                [&](GenomeSweepPacked const& seq) {
                    return run_var_bytewise_simd_butterfly_avx512_vectors(
                        seq, bf_avx512_kernel, sink_buffer
                    );
                }
            ),
            bench(
                "bytewise_simd_butterfly_avx512_scalars",
                [&](GenomeSweepPacked const& seq) {
                    return run_var_bytewise_simd_butterfly_avx512_scalars(
                        seq, bf_avx512_kernel, sink_buffer
                    );
                }
            ),
            #endif
            #if defined(FISK_HAS_NEON)
            bench(
                "bytewise_simd_butterfly_neon_vectors",
                [&](GenomeSweepPacked const& seq) {
                    return run_var_bytewise_simd_butterfly_neon_vectors(
                        seq, bf_neon_kernel, sink_buffer
                    );
                }
            ),
            bench(
                "bytewise_simd_butterfly_neon_scalars",
                [&](GenomeSweepPacked const& seq) {
                    return run_var_bytewise_simd_butterfly_neon_scalars(
                        seq, bf_neon_kernel, sink_buffer
                    );
                }
            ),
            #endif
            bench(
                "bytewise_scalar_butterfly",
                [&](GenomeSweepPacked const& seq) {
                    return run_var_bytewise_scalar_butterfly(
                        seq, bit_ext_butterfly_mask, sink_buffer
                    );
                }
            )
        );
        std::string const estimate = packed.empty()
            ? std::string()
            : for_each_spaced_kmer_algorithm_name(packed.front(), mask_forms);
        write_csv_rows(
            csv_os, suite_title, "input=packed;kind=spaced_kmers;estimate=" + estimate, results,
            kSinkName
        );
    }
}
