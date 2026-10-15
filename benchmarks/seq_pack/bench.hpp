#pragma once

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/seq_pack/seq_pack.hpp"
#include "fisk/core/simd_sse2.hpp"
#include "fisk/core/simd_avx2.hpp"
#include "fisk/core/simd_avx512.hpp"
#include "fisk/core/simd_neon.hpp"

/**
 * @brief Benchmark pack_sequence() across all four encoding x layout combinations.
 *
 * Each combination (encoding=acgt/actg, layout=lsb/msb) runs as its own comparison group: within
 * one group, all candidates are supposed to agree bit-for-bit (same encoding,
 * same layout), which is exactly what the sink cross-validation checks.
 */
void bench_seq_pack(std::vector<std::string> const& sequences, std::ostream& csv_os);

// Kernels compared above, each defined in its own translation unit (var_*.cpp). Each one packs
// into `out`, owned and reused by the calling suite (see bench.cpp) rather than by the kernel
// itself, so that summing it into a sink can happen once, in Microbench::finalize(), outside the
// timed region. Guarded to match the ISA availability of the word encoder each one benchmarks
// (see fisk/core/intrinsics.hpp).
std::uint64_t run_var_acgt_lsb_butterfly_scalar(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_butterfly_scalar(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_butterfly_scalar(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_butterfly_scalar(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_acgt_lsb_fold_scalar(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_fold_scalar(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_fold_scalar(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_fold_scalar(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_acgt_lsb_direct_scalar(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_direct_scalar(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_direct_scalar(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_direct_scalar(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);

#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_acgt_lsb_pext(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_pext(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_pext(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_pext(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_acgt_lsb_pext_sse2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_pext_sse2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_pext_sse2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_pext_sse2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif
#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_acgt_lsb_pext_avx2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_pext_avx2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_pext_avx2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_pext_avx2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif
#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_acgt_lsb_pext_avx512(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_pext_avx512(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_pext_avx512(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_pext_avx512(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif
#endif

#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_acgt_lsb_butterfly_sse2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_butterfly_sse2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_butterfly_sse2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_butterfly_sse2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_acgt_lsb_fold_sse2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_fold_sse2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_fold_sse2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_fold_sse2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_acgt_lsb_direct_sse2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_direct_sse2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_direct_sse2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_direct_sse2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#if defined(FISK_HAS_SSSE3)
std::uint64_t run_var_acgt_lsb_direct_double_sse2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_direct_double_sse2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_direct_double_sse2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_direct_double_sse2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif
#endif

#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_acgt_lsb_butterfly_avx2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_butterfly_avx2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_butterfly_avx2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_butterfly_avx2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_acgt_lsb_fold_avx2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_fold_avx2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_fold_avx2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_fold_avx2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_acgt_lsb_direct_avx2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_direct_avx2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_direct_avx2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_direct_avx2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_acgt_lsb_direct_double_avx2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_direct_double_avx2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_direct_double_avx2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_direct_double_avx2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif

#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_acgt_lsb_butterfly_avx512(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_butterfly_avx512(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_butterfly_avx512(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_butterfly_avx512(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_acgt_lsb_fold_avx512(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_fold_avx512(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_fold_avx512(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_fold_avx512(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_acgt_lsb_direct_avx512(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_direct_avx512(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_direct_avx512(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_direct_avx512(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_acgt_lsb_direct_double_avx512(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_direct_double_avx512(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_direct_double_avx512(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_direct_double_avx512(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif

#if defined(FISK_HAS_NEON)
std::uint64_t run_var_acgt_lsb_butterfly_neon(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_butterfly_neon(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_butterfly_neon(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_butterfly_neon(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_acgt_lsb_fold_neon(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_fold_neon(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_fold_neon(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_fold_neon(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_acgt_lsb_direct_neon(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_direct_neon(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_direct_neon(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_direct_neon(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_acgt_lsb_direct_double_neon(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_direct_double_neon(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_direct_double_neon(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_direct_double_neon(
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
