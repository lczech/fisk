#pragma once

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/simd/sse2.hpp"
#include "fisk/simd/avx2.hpp"
#include "fisk/simd/avx512.hpp"
#include "fisk/simd/neon.hpp"

/**
 * @brief Benchmark pack_sequence_unchecked() across all four encoding x layout combinations.
 *
 * Each combination (encoding=acgt/actg, layout=lsb/msb) runs as its own comparison group: within
 * one group, all candidates are supposed to agree bit-for-bit (same encoding,
 * same layout), which is exactly what the sink cross-validation checks.
 */
void bench_sequence_pack_unchecked(std::vector<std::string> const& sequences, std::ostream& csv_os);

// Kernels compared above, each defined in its own translation unit (var_*.cpp). Each one packs
// into `out`, owned and reused by the calling suite (see bench.cpp) rather than by the kernel
// itself, so that summing it into a sink can happen once, in Microbench::finalize(), outside the
// timed region. Guarded to match the ISA availability of the word encoder each one benchmarks
// (see fisk/util/intrinsics.hpp).
std::uint64_t run_var_unchecked_butterfly_scalar_acgt_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_butterfly_scalar_acgt_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_butterfly_scalar_actg_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_butterfly_scalar_actg_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_fold_scalar_acgt_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_fold_scalar_acgt_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_fold_scalar_actg_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_fold_scalar_actg_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_direct_scalar_acgt_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_direct_scalar_acgt_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_direct_scalar_actg_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_direct_scalar_actg_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);

#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_unchecked_pext_scalar_acgt_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_pext_scalar_acgt_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_pext_scalar_actg_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_pext_scalar_actg_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_unchecked_pext_sse2_acgt_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_pext_sse2_acgt_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_pext_sse2_actg_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_pext_sse2_actg_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif
#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_unchecked_pext_avx2_acgt_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_pext_avx2_acgt_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_pext_avx2_actg_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_pext_avx2_actg_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif
#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_unchecked_pext_avx512_acgt_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_pext_avx512_acgt_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_pext_avx512_actg_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_pext_avx512_actg_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif
#endif

#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_unchecked_butterfly_sse2_acgt_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_butterfly_sse2_acgt_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_butterfly_sse2_actg_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_butterfly_sse2_actg_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_fold_sse2_acgt_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_fold_sse2_acgt_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_fold_sse2_actg_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_fold_sse2_actg_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_direct_sse2_acgt_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_direct_sse2_acgt_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_direct_sse2_actg_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_direct_sse2_actg_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#if defined(FISK_HAS_SSSE3)
std::uint64_t run_var_unchecked_direct_double_sse2_acgt_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_direct_double_sse2_acgt_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_direct_double_sse2_actg_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_direct_double_sse2_actg_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif
#endif

#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_unchecked_butterfly_avx2_acgt_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_butterfly_avx2_acgt_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_butterfly_avx2_actg_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_butterfly_avx2_actg_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_fold_avx2_acgt_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_fold_avx2_acgt_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_fold_avx2_actg_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_fold_avx2_actg_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_direct_avx2_acgt_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_direct_avx2_acgt_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_direct_avx2_actg_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_direct_avx2_actg_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_direct_double_avx2_acgt_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_direct_double_avx2_acgt_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_direct_double_avx2_actg_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_direct_double_avx2_actg_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif

#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_unchecked_butterfly_avx512_acgt_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_butterfly_avx512_acgt_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_butterfly_avx512_actg_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_butterfly_avx512_actg_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_fold_avx512_acgt_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_fold_avx512_acgt_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_fold_avx512_actg_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_fold_avx512_actg_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_direct_avx512_acgt_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_direct_avx512_acgt_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_direct_avx512_actg_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_direct_avx512_actg_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_direct_double_avx512_acgt_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_direct_double_avx512_acgt_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_direct_double_avx512_actg_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_direct_double_avx512_actg_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif

#if defined(FISK_HAS_NEON)
std::uint64_t run_var_unchecked_butterfly_neon_acgt_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_butterfly_neon_acgt_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_butterfly_neon_actg_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_butterfly_neon_actg_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_fold_neon_acgt_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_fold_neon_acgt_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_fold_neon_actg_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_fold_neon_actg_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_direct_neon_acgt_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_direct_neon_acgt_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_direct_neon_actg_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_direct_neon_actg_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_direct_double_neon_acgt_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_direct_double_neon_acgt_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_unchecked_direct_double_neon_actg_lsb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_unchecked_direct_double_neon_actg_msb(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif

// =================================================================================================
//     FastDNA References
// =================================================================================================

/**
 * @brief Adapted paper kernels; see external/fastdna.hpp for the format adjustments.
 */
std::uint64_t run_var_fastdna_lut(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);

#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_fastdna_pext_ordered(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_fastdna_pext_unordered(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_fastdna_pext_unordered_unroll2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_fastdna_pext_unordered_unroll4(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);

#endif // FISK_HAS_BMI2

// =================================================================================================
//     cute-nucleotides References
// =================================================================================================

/**
 * @brief Adapted kernels; see external/cute_nucleotides.hpp for the format adjustments, and for how
 * their other algorithms map to the word encoders of fisk.
 */
#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_cute_mul(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_cute_movemask(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
#endif // FISK_HAS_AVX2
