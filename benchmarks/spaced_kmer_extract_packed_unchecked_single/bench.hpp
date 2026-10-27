#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/spaced_kmer_extract/packed_bytewise_scalar.hpp"
#include "fisk/spaced_kmer_extract/packed_bytewise_simd.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/simd/scalar.hpp"
#include "fisk/simd/sse2.hpp"
#include "fisk/simd/avx2.hpp"
#include "fisk/simd/avx512.hpp"
#include "fisk/simd/neon.hpp"
#include "sink.hpp"

using PackedMsb = fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>;
using PackedLsb = fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>;

/**
 * @brief Benchmark spaced k-mer extract from packed sequences with single masks.
 */
void bench_spaced_kmer_extract_packed_unchecked_single(
    std::vector<std::string> const& sequences,
    std::vector<std::string> const& masks,
    std::ostream& csv_os
);

// Kernels compared above, each defined in its own translation unit (var_*.cpp). Each one takes,
// alongside the packed sequence, the piece of per-mask setup it needs (which carries the span k) --
// prepared once per mask in bench.cpp and threaded in here, so that setup cost never enters the
// timed region. Guarded to match the ISA availability of the kernel each one benchmarks (see
// fisk/util/intrinsics.hpp). They are overloaded for both Layouts, defined in var_*_msb.cpp and
// var_*_lsb.cpp, respectively. The SIMD ones take their spaced k-mers either as vectors
// (`_vectors`), or one at a time (`_scalars`).
std::uint64_t run_var_bytewise_scalar_bit_loop(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_bit_loop(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_byte_lookup(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_byte_lookup(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_bytewise_scalar_pext(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_pext(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
std::uint64_t run_var_bytewise_scalar_block_shift(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockShiftMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_block_shift(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockShiftMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_block_shift_unrolled2(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockShiftMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_block_shift_unrolled2(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockShiftMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_block_shift_unrolled4(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockShiftMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_block_shift_unrolled4(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockShiftMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_block_shift_unrolled8(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockShiftMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_block_shift_unrolled8(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockShiftMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_butterfly(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractButterflyMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_butterfly(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractButterflyMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);

#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_bytewise_simd_butterfly_sse2_vectors(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_sse2_vectors(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_sse2_scalars(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_sse2_scalars(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_sse2_vectors(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdSSE2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_sse2_vectors(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdSSE2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_sse2_scalars(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdSSE2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_sse2_scalars(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdSSE2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_bytewise_simd_butterfly_avx2_vectors(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_avx2_vectors(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_avx2_scalars(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_avx2_scalars(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_avx2_vectors(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_avx2_vectors(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_avx2_scalars(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_avx2_scalars(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_bytewise_simd_butterfly_avx512_vectors(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_avx512_vectors(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_avx512_scalars(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_avx512_scalars(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_avx512_vectors(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX512>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_avx512_vectors(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX512>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_avx512_scalars(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX512>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_avx512_scalars(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX512>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
#if defined(FISK_HAS_NEON)
std::uint64_t run_var_bytewise_simd_butterfly_neon_vectors(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_neon_vectors(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_neon_scalars(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_neon_scalars(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_neon_vectors(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdNEON>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_neon_vectors(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdNEON>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_neon_scalars(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdNEON>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_neon_scalars(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdNEON>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_bytewise_simd_pext_vectors(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelPext> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_pext_vectors(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelPext> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_pext_scalars(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelPext> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_pext_scalars(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelPext> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
std::uint64_t run_var_bytewise_simd_butterfly_scalar_vectors(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_scalar_vectors(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_scalar_scalars(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_scalar_scalars(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_scalar_vectors(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdScalar>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_scalar_vectors(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdScalar>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_scalar_scalars(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdScalar>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_scalar_scalars(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdScalar>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
