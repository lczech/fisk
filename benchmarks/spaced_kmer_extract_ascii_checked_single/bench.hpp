#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/spaced_kmer_extract/ascii_rolling_simd.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"
#include "fisk/core/kmer_spaced.hpp"
#include "fisk/simd/scalar.hpp"
#include "fisk/simd/sse2.hpp"
#include "fisk/simd/avx2.hpp"
#include "fisk/simd/avx512.hpp"
#include "fisk/simd/neon.hpp"
#include "chunked.hpp"
#include "seq_data.hpp"
#include "sink.hpp"

/**
 * @brief Benchmark spaced k-mer extract with single masks.
 */
void bench_spaced_kmer_extract_ascii_checked_single(
    std::vector<InputSequence> const& sequences,
    std::vector<std::string> const& masks,
    std::ostream& csv_os
);

// Kernels compared above, each defined in its own translation unit (var_*.cpp). Each one takes,
// alongside the sequence, the piece of per-mask setup it needs (which carries the span k) --
// prepared once per mask in bench.cpp and threaded in here, so that setup cost never enters the
// timed region. Guarded to match the ISA availability of the kernel each one benchmarks (see
// fisk/util/intrinsics.hpp). The ones taking spaced k-mer masks are overloaded for both Layouts,
// defined in var_*_msb.cpp and var_*_lsb.cpp, respectively.
//
// The SIMD ones take their spaced k-mers either as vectors (`_vectors`), or one at a time with
// for_each_valid_lane() (`_scalars`). The `_vectors` ones consume only their valid lanes, see
// consume_valid_lanes() (sink.hpp), so that all variants sum the same spaced k-mers and share one
// suite.run().
std::uint64_t run_var_missh(
    InputSequence const& seq,
    std::size_t k,
    std::vector<std::size_t> const& naive_mask,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_naive(
    InputSequence const& seq,
    std::size_t k,
    std::vector<std::size_t> const& naive_mask,
    SinkBuffer& sink_buffer
);

std::uint64_t run_var_rolling_scalar_bit_loop(
    InputSequence const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask> const& mask,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_scalar_bit_loop(
    InputSequence const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask> const& mask,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_scalar_byte_lookup(
    InputSequence const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask> const& mask,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_scalar_byte_lookup(
    InputSequence const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask> const& mask,
    SinkBuffer& sink_buffer
);
#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_rolling_scalar_pext(
    InputSequence const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask> const& mask,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_scalar_pext(
    InputSequence const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask> const& mask,
    SinkBuffer& sink_buffer
);
#endif

std::uint64_t run_var_rolling_scalar_block_shift(
    InputSequence const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockShiftMask> const& mask,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_scalar_block_shift(
    InputSequence const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockShiftMask> const& mask,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_scalar_block_shift_unrolled2(
    InputSequence const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockShiftMask> const& mask,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_scalar_block_shift_unrolled2(
    InputSequence const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockShiftMask> const& mask,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_scalar_block_shift_unrolled4(
    InputSequence const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockShiftMask> const& mask,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_scalar_block_shift_unrolled4(
    InputSequence const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockShiftMask> const& mask,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_scalar_block_shift_unrolled8(
    InputSequence const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockShiftMask> const& mask,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_scalar_block_shift_unrolled8(
    InputSequence const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockShiftMask> const& mask,
    SinkBuffer& sink_buffer
);

std::uint64_t run_var_rolling_scalar_butterfly(
    InputSequence const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractButterflyMask> const& mask,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_scalar_butterfly(
    InputSequence const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractButterflyMask> const& mask,
    SinkBuffer& sink_buffer
);

std::uint64_t run_var_rolling_simd_butterfly_scalar_vectors(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_scalar_scalars(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_scalar_vectors(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_scalar_scalars(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_scalar_vectors(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdScalar>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_scalar_scalars(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdScalar>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_scalar_vectors(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdScalar>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_scalar_scalars(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdScalar>
    > const& kernel,
    SinkBuffer& sink_buffer
);

#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_rolling_simd_butterfly_sse2_vectors(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_sse2_scalars(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_sse2_vectors(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_sse2_scalars(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_sse2_vectors(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdSSE2>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_sse2_scalars(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdSSE2>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_sse2_vectors(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdSSE2>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_sse2_scalars(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdSSE2>
    > const& kernel,
    SinkBuffer& sink_buffer
);
#endif

#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_rolling_simd_butterfly_avx2_vectors(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_avx2_scalars(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_avx2_vectors(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_avx2_scalars(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_avx2_vectors(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX2>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_avx2_scalars(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX2>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_avx2_vectors(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX2>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_avx2_scalars(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX2>
    > const& kernel,
    SinkBuffer& sink_buffer
);
#endif

#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_rolling_simd_butterfly_avx512_vectors(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_avx512_scalars(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_avx512_vectors(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_avx512_scalars(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_avx512_vectors(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX512>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_avx512_scalars(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX512>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_avx512_vectors(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX512>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_avx512_scalars(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX512>
    > const& kernel,
    SinkBuffer& sink_buffer
);
#endif

#if defined(FISK_HAS_NEON)
std::uint64_t run_var_rolling_simd_butterfly_neon_vectors(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_neon_scalars(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_neon_vectors(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_neon_scalars(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_neon_vectors(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdNEON>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_neon_scalars(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdNEON>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_neon_vectors(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdNEON>
    > const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_neon_scalars(
    InputSequence const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdNEON>
    > const& kernel,
    SinkBuffer& sink_buffer
);
#endif

#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_rolling_simd_pext_vectors(
    InputSequence const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelPext> const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_pext_scalars(
    InputSequence const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelPext> const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_pext_vectors(
    InputSequence const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelPext> const& kernel,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_rolling_simd_pext_scalars(
    InputSequence const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelPext> const& kernel,
    SinkBuffer& sink_buffer
);
#endif

// The chunked variants, see spaced_kmer_extract/ascii_chunked_scalar.hpp, each with a scratch
// of its own. The `_scalars` SIMD variants take the spaced k-mers out of the vectors with
// for_each_valid_lane(), the `_vectors` ones consume only their valid lanes, see
// consume_valid_lanes() (sink.hpp), so that their sums match the others.
#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_chunked_scalar_pext(
    InputSequence const& seq,
    PextMask<fisk::Layout::kMSB> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kMSB>& scratch,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_chunked_scalar_pext(
    InputSequence const& seq,
    PextMask<fisk::Layout::kLSB> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kLSB>& scratch,
    SinkBuffer& sink_buffer
);
#endif
std::uint64_t run_var_chunked_scalar_butterfly(
    InputSequence const& seq,
    ButterflyMask<fisk::Layout::kMSB> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kMSB>& scratch,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_chunked_scalar_butterfly(
    InputSequence const& seq,
    ButterflyMask<fisk::Layout::kLSB> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kLSB>& scratch,
    SinkBuffer& sink_buffer
);
#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_chunked_simd_butterfly_sse2_scalars(
    InputSequence const& seq,
    ButterflyKernel<fisk::Layout::kMSB, fisk::SimdSSE2> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kMSB>& scratch,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_chunked_simd_butterfly_sse2_scalars(
    InputSequence const& seq,
    ButterflyKernel<fisk::Layout::kLSB, fisk::SimdSSE2> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kLSB>& scratch,
    SinkBuffer& sink_buffer
);
#endif
#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_chunked_simd_butterfly_avx2_scalars(
    InputSequence const& seq,
    ButterflyKernel<fisk::Layout::kMSB, fisk::SimdAVX2> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kMSB>& scratch,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_chunked_simd_butterfly_avx2_scalars(
    InputSequence const& seq,
    ButterflyKernel<fisk::Layout::kLSB, fisk::SimdAVX2> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kLSB>& scratch,
    SinkBuffer& sink_buffer
);
#endif
#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_chunked_simd_butterfly_avx512_scalars(
    InputSequence const& seq,
    ButterflyKernel<fisk::Layout::kMSB, fisk::SimdAVX512> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kMSB>& scratch,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_chunked_simd_butterfly_avx512_scalars(
    InputSequence const& seq,
    ButterflyKernel<fisk::Layout::kLSB, fisk::SimdAVX512> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kLSB>& scratch,
    SinkBuffer& sink_buffer
);
#endif
#if defined(FISK_HAS_NEON)
std::uint64_t run_var_chunked_simd_butterfly_neon_scalars(
    InputSequence const& seq,
    ButterflyKernel<fisk::Layout::kMSB, fisk::SimdNEON> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kMSB>& scratch,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_chunked_simd_butterfly_neon_scalars(
    InputSequence const& seq,
    ButterflyKernel<fisk::Layout::kLSB, fisk::SimdNEON> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kLSB>& scratch,
    SinkBuffer& sink_buffer
);
#endif
std::uint64_t run_var_chunked_simd_butterfly_scalar_scalars(
    InputSequence const& seq,
    ButterflyKernel<fisk::Layout::kMSB, fisk::SimdScalar> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kMSB>& scratch,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_chunked_simd_butterfly_scalar_scalars(
    InputSequence const& seq,
    ButterflyKernel<fisk::Layout::kLSB, fisk::SimdScalar> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kLSB>& scratch,
    SinkBuffer& sink_buffer
);
#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_chunked_simd_butterfly_sse2_vectors(
    InputSequence const& seq,
    ButterflyKernel<fisk::Layout::kMSB, fisk::SimdSSE2> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kMSB>& scratch,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_chunked_simd_butterfly_sse2_vectors(
    InputSequence const& seq,
    ButterflyKernel<fisk::Layout::kLSB, fisk::SimdSSE2> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kLSB>& scratch,
    SinkBuffer& sink_buffer
);
#endif
#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_chunked_simd_butterfly_avx2_vectors(
    InputSequence const& seq,
    ButterflyKernel<fisk::Layout::kMSB, fisk::SimdAVX2> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kMSB>& scratch,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_chunked_simd_butterfly_avx2_vectors(
    InputSequence const& seq,
    ButterflyKernel<fisk::Layout::kLSB, fisk::SimdAVX2> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kLSB>& scratch,
    SinkBuffer& sink_buffer
);
#endif
#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_chunked_simd_butterfly_avx512_vectors(
    InputSequence const& seq,
    ButterflyKernel<fisk::Layout::kMSB, fisk::SimdAVX512> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kMSB>& scratch,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_chunked_simd_butterfly_avx512_vectors(
    InputSequence const& seq,
    ButterflyKernel<fisk::Layout::kLSB, fisk::SimdAVX512> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kLSB>& scratch,
    SinkBuffer& sink_buffer
);
#endif
#if defined(FISK_HAS_NEON)
std::uint64_t run_var_chunked_simd_butterfly_neon_vectors(
    InputSequence const& seq,
    ButterflyKernel<fisk::Layout::kMSB, fisk::SimdNEON> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kMSB>& scratch,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_chunked_simd_butterfly_neon_vectors(
    InputSequence const& seq,
    ButterflyKernel<fisk::Layout::kLSB, fisk::SimdNEON> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kLSB>& scratch,
    SinkBuffer& sink_buffer
);
#endif
std::uint64_t run_var_chunked_simd_butterfly_scalar_vectors(
    InputSequence const& seq,
    ButterflyKernel<fisk::Layout::kMSB, fisk::SimdScalar> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kMSB>& scratch,
    SinkBuffer& sink_buffer
);
std::uint64_t run_var_chunked_simd_butterfly_scalar_vectors(
    InputSequence const& seq,
    ButterflyKernel<fisk::Layout::kLSB, fisk::SimdScalar> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kLSB>& scratch,
    SinkBuffer& sink_buffer
);