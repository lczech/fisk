#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/kmer_spaced/packed.hpp"
#include "fisk/kmer_spaced/packed_simd.hpp"
#include "fisk/core/simd_scalar.hpp"
#include "fisk/core/simd_sse2.hpp"
#include "fisk/core/simd_avx2.hpp"
#include "fisk/core/simd_avx512.hpp"
#include "fisk/core/simd_neon.hpp"
#include "fisk/seq_pack/seq_pack.hpp"
#include "sink.hpp"

// Every sequence benchmarked below is packed with its validity, with an ACGT word encoder; only
// the layout varies.
using CheckedMsb = fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>;
using CheckedLsb = fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>;

// The variants that hand out vectors sum them whole, valid lanes or not, as a caller takes the
// vectors with their valid lanes as they are, and keep the valid lanes with a do_not_optimize()
// barrier, so that they are computed. The sums of the vector variants hence only match within an
// arch, see bench_kmer_spaced_packed_checked_multi().

/**
 * @brief Benchmark checked spaced k-mer extraction from a PackedSequenceWithValidity with
 * multiple masks at once, the same variants as bench_kmer_spaced_packed_multi(), each extracting
 * only the valid spaced k-mers.
 *
 * Packing with validity is done once per sequence, outside the timed region. Which nucleotides are
 * invalid, and whether they come in runs, depends on the input.
 */
void bench_kmer_spaced_packed_checked_multi(
    std::vector<std::string> const& sequences,
    std::vector<std::vector<std::string>> const& multi_masks,
    std::ostream& csv_os
);

// Kernels compared above, each defined in its own translation unit (var_*.cpp). Each one takes,
// alongside the packed sequence, the piece of per-mask-set setup it needs (which carries the span
// k) -- prepared once per mask set in bench.cpp and threaded in here, so that setup cost never
// enters the timed region. Guarded to match the ISA availability of the kernel each one benchmarks
// (see fisk/core/intrinsics.hpp). They are overloaded for both Layouts, defined in var_msb_*.cpp
// and var_lsb_*.cpp, respectively. The SIMD ones take their spaced k-mers either as vectors
// (`_vectors`), or one at a time (`_scalars`).
std::uint64_t run_var_bitloop(
    CheckedMsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bitloop(
    CheckedLsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_byte_table(
    CheckedMsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_byte_table(
    CheckedLsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_pext(
    CheckedMsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_pext(
    CheckedLsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
std::uint64_t run_var_block_table(
    CheckedMsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table(
    CheckedLsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled2(
    CheckedMsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled2(
    CheckedLsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled4(
    CheckedMsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled4(
    CheckedLsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled8(
    CheckedMsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled8(
    CheckedLsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_butterfly_table(
    CheckedMsb const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractButterflyTable>
    > const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_butterfly_table(
    CheckedLsb const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractButterflyTable>
    > const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_simd_butterfly_table_sse2_vectors(
    CheckedMsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_sse2_vectors(
    CheckedLsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_sse2_scalars(
    CheckedMsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_sse2_scalars(
    CheckedLsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_sse2_vectors(
    CheckedMsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdSSE2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_sse2_vectors(
    CheckedLsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdSSE2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_sse2_scalars(
    CheckedMsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdSSE2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_sse2_scalars(
    CheckedLsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdSSE2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_simd_butterfly_table_avx2_vectors(
    CheckedMsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_avx2_vectors(
    CheckedLsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_avx2_scalars(
    CheckedMsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_avx2_scalars(
    CheckedLsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_avx2_vectors(
    CheckedMsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdAVX2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_avx2_vectors(
    CheckedLsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdAVX2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_avx2_scalars(
    CheckedMsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdAVX2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_avx2_scalars(
    CheckedLsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdAVX2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_simd_butterfly_table_avx512_vectors(
    CheckedMsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_avx512_vectors(
    CheckedLsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_avx512_scalars(
    CheckedMsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_avx512_scalars(
    CheckedLsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_avx512_vectors(
    CheckedMsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdAVX512>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_avx512_vectors(
    CheckedLsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdAVX512>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_avx512_scalars(
    CheckedMsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdAVX512>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_avx512_scalars(
    CheckedLsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdAVX512>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
#if defined(FISK_HAS_NEON)
std::uint64_t run_var_simd_butterfly_table_neon_vectors(
    CheckedMsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_neon_vectors(
    CheckedLsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_neon_scalars(
    CheckedMsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_neon_scalars(
    CheckedLsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_neon_vectors(
    CheckedMsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdNEON>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_neon_vectors(
    CheckedLsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdNEON>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_neon_scalars(
    CheckedMsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdNEON>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_neon_scalars(
    CheckedLsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdNEON>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_simd_pext_vectors(
    CheckedMsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelPEXT>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_pext_vectors(
    CheckedLsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelPEXT>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_pext_scalars(
    CheckedMsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelPEXT>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_pext_scalars(
    CheckedLsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelPEXT>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
std::uint64_t run_var_simd_butterfly_table_scalar_vectors(
    CheckedMsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_scalar_vectors(
    CheckedLsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_scalar_scalars(
    CheckedMsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_scalar_scalars(
    CheckedLsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_scalar_vectors(
    CheckedMsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdScalar>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_scalar_vectors(
    CheckedLsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdScalar>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_scalar_scalars(
    CheckedMsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdScalar>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_scalar_scalars(
    CheckedLsb const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdScalar>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
