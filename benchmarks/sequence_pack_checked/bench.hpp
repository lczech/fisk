#pragma once

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/core/simd_sse2.hpp"
#include "fisk/core/simd_avx2.hpp"
#include "fisk/core/simd_avx512.hpp"
#include "fisk/core/simd_neon.hpp"

/**
 * @brief Benchmark pack_sequence_checked() across all four encoding x layout combinations.
 *
 * The same variants as bench_sequence_pack_unchecked(), each packing into a
 * PackedSequenceWithValidity instead, so that the cost of the validity bits is the difference to
 * the matching variant there.
 */
void bench_sequence_pack_checked(std::vector<std::string> const& sequences, std::ostream& csv_os);

// Kernels compared above, each defined in its own translation unit (var_*.cpp), and guarded to
// match the ISA availability of its word encoder, as in sequence_pack_unchecked/bench.hpp.

std::uint64_t run_var_checked_butterfly_scalar_acgt_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_butterfly_scalar_acgt_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_butterfly_scalar_actg_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_butterfly_scalar_actg_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_fold_scalar_acgt_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_fold_scalar_acgt_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_fold_scalar_actg_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_fold_scalar_actg_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_direct_scalar_acgt_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_direct_scalar_acgt_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_direct_scalar_actg_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_direct_scalar_actg_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);

#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_checked_pext_scalar_acgt_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_pext_scalar_acgt_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_pext_scalar_actg_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_pext_scalar_actg_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_checked_pext_sse2_acgt_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_pext_sse2_acgt_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_pext_sse2_actg_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_pext_sse2_actg_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif
#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_checked_pext_avx2_acgt_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_pext_avx2_acgt_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_pext_avx2_actg_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_pext_avx2_actg_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif
#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_checked_pext_avx512_acgt_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_pext_avx512_acgt_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_pext_avx512_actg_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_pext_avx512_actg_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif
#endif

#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_checked_butterfly_sse2_acgt_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_butterfly_sse2_acgt_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_butterfly_sse2_actg_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_butterfly_sse2_actg_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_fold_sse2_acgt_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_fold_sse2_acgt_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_fold_sse2_actg_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_fold_sse2_actg_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_direct_sse2_acgt_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_direct_sse2_acgt_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_direct_sse2_actg_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_direct_sse2_actg_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#if defined(FISK_HAS_SSSE3)
std::uint64_t run_var_checked_direct_double_sse2_acgt_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_direct_double_sse2_acgt_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_direct_double_sse2_actg_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_direct_double_sse2_actg_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif
#endif

#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_checked_butterfly_avx2_acgt_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_butterfly_avx2_acgt_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_butterfly_avx2_actg_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_butterfly_avx2_actg_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_fold_avx2_acgt_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_fold_avx2_acgt_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_fold_avx2_actg_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_fold_avx2_actg_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_direct_avx2_acgt_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_direct_avx2_acgt_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_direct_avx2_actg_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_direct_avx2_actg_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_direct_double_avx2_acgt_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_direct_double_avx2_acgt_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_direct_double_avx2_actg_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_direct_double_avx2_actg_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif

#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_checked_butterfly_avx512_acgt_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_butterfly_avx512_acgt_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_butterfly_avx512_actg_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_butterfly_avx512_actg_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_fold_avx512_acgt_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_fold_avx512_acgt_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_fold_avx512_actg_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_fold_avx512_actg_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_direct_avx512_acgt_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_direct_avx512_acgt_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_direct_avx512_actg_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_direct_avx512_actg_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_direct_double_avx512_acgt_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_direct_double_avx512_acgt_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_direct_double_avx512_actg_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_direct_double_avx512_actg_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif

#if defined(FISK_HAS_NEON)
std::uint64_t run_var_checked_butterfly_neon_acgt_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_butterfly_neon_acgt_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_butterfly_neon_actg_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_butterfly_neon_actg_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_fold_neon_acgt_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_fold_neon_acgt_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_fold_neon_actg_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_fold_neon_actg_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_direct_neon_acgt_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_direct_neon_acgt_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_direct_neon_actg_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_direct_neon_actg_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_direct_double_neon_acgt_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_direct_double_neon_acgt_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_checked_direct_double_neon_actg_lsb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_checked_direct_double_neon_actg_msb(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif
