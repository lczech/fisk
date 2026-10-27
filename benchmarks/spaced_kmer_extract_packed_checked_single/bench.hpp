#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/spaced_kmer_extract/packed_bytewise_scalar.hpp"
#include "fisk/spaced_kmer_extract/packed_bytewise_simd.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"
#include "fisk/simd/scalar.hpp"
#include "fisk/simd/sse2.hpp"
#include "fisk/simd/avx2.hpp"
#include "fisk/simd/avx512.hpp"
#include "fisk/simd/neon.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "sink.hpp"

// Every sequence benchmarked below is packed with its validity, with an ACGT word encoder; only
// the layout varies.
using CheckedMsb = fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>;
using CheckedLsb = fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>;

// The variants that hand out vectors sum them whole, valid lanes or not, as a caller takes the
// vectors with their valid lanes as they are, and keep the valid lanes with a do_not_optimize()
// barrier, so that they are computed. The sums of the vector variants hence only match within an
// arch, see bench_spaced_kmer_extract_packed_checked_single().

/**
 * @brief Benchmark checked spaced k-mer extraction from a PackedSequenceWithValidity with
 * single masks, the same variants as bench_spaced_kmer_extract_packed_unchecked_single(), each
 * extracting only the valid spaced k-mers.
 *
 * Packing with validity is done once per sequence, outside the timed region. Which nucleotides are
 * invalid, and whether they come in runs, depends on the input.
 */
void bench_spaced_kmer_extract_packed_checked_single(
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
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_bit_loop(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_byte_lookup(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_byte_lookup(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_bytewise_scalar_pext(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_pext(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
std::uint64_t run_var_bytewise_scalar_block_shift(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockShiftMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_block_shift(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockShiftMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_block_shift_unrolled2(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockShiftMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_block_shift_unrolled2(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockShiftMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_block_shift_unrolled4(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockShiftMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_block_shift_unrolled4(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockShiftMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_block_shift_unrolled8(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockShiftMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_block_shift_unrolled8(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockShiftMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_butterfly(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractButterflyMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_butterfly(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractButterflyMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_bytewise_simd_butterfly_sse2_vectors(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_sse2_vectors(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_sse2_scalars(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_sse2_scalars(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_sse2_vectors(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdSSE2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_sse2_vectors(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdSSE2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_sse2_scalars(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdSSE2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_sse2_scalars(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdSSE2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_bytewise_simd_butterfly_avx2_vectors(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_avx2_vectors(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_avx2_scalars(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_avx2_scalars(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_avx2_vectors(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_avx2_vectors(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_avx2_scalars(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_avx2_scalars(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_bytewise_simd_butterfly_avx512_vectors(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_avx512_vectors(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_avx512_scalars(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_avx512_scalars(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_avx512_vectors(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX512>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_avx512_vectors(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX512>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_avx512_scalars(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX512>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_avx512_scalars(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX512>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
#if defined(FISK_HAS_NEON)
std::uint64_t run_var_bytewise_simd_butterfly_neon_vectors(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_neon_vectors(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_neon_scalars(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_neon_scalars(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_neon_vectors(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdNEON>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_neon_vectors(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdNEON>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_neon_scalars(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdNEON>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_neon_scalars(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdNEON>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_bytewise_simd_pext_vectors(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelPext> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_pext_vectors(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelPext> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_pext_scalars(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelPext> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_pext_scalars(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelPext> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
std::uint64_t run_var_bytewise_simd_butterfly_scalar_vectors(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_scalar_vectors(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_scalar_scalars(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_scalar_scalars(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_scalar_vectors(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdScalar>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_scalar_vectors(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdScalar>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_scalar_scalars(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdScalar>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_block_shift_scalar_scalars(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdScalar>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
