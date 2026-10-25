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
 * The same variants as bench_seq_pack(), each packing into a PackedSequenceWithValidity instead,
 * so that the cost of the validity bits is the difference to the matching variant there.
 */
void bench_seq_pack_validity(std::vector<std::string> const& sequences, std::ostream& csv_os);

// Kernels compared above, each defined in its own translation unit (var_*.cpp), and guarded to
// match the ISA availability of its word encoder, as in seq_pack/bench.hpp.

std::uint64_t run_var_validity_acgt_lsb_butterfly_scalar(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_acgt_msb_butterfly_scalar(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_actg_lsb_butterfly_scalar(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_actg_msb_butterfly_scalar(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_acgt_lsb_fold_scalar(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_acgt_msb_fold_scalar(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_actg_lsb_fold_scalar(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_actg_msb_fold_scalar(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_acgt_lsb_direct_scalar(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_acgt_msb_direct_scalar(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_actg_lsb_direct_scalar(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_actg_msb_direct_scalar(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);

#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_validity_acgt_lsb_pext(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_acgt_msb_pext(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_actg_lsb_pext(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_actg_msb_pext(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_validity_acgt_lsb_pext_sse2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_acgt_msb_pext_sse2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_actg_lsb_pext_sse2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_actg_msb_pext_sse2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif
#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_validity_acgt_lsb_pext_avx2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_acgt_msb_pext_avx2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_actg_lsb_pext_avx2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_actg_msb_pext_avx2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif
#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_validity_acgt_lsb_pext_avx512(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_acgt_msb_pext_avx512(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_actg_lsb_pext_avx512(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_actg_msb_pext_avx512(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif
#endif

#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_validity_acgt_lsb_butterfly_sse2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_acgt_msb_butterfly_sse2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_actg_lsb_butterfly_sse2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_actg_msb_butterfly_sse2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_acgt_lsb_fold_sse2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_acgt_msb_fold_sse2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_actg_lsb_fold_sse2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_actg_msb_fold_sse2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_acgt_lsb_direct_sse2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_acgt_msb_direct_sse2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_actg_lsb_direct_sse2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_actg_msb_direct_sse2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#if defined(FISK_HAS_SSSE3)
std::uint64_t run_var_validity_acgt_lsb_direct_double_sse2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_acgt_msb_direct_double_sse2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_actg_lsb_direct_double_sse2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_actg_msb_direct_double_sse2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif
#endif

#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_validity_acgt_lsb_butterfly_avx2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_acgt_msb_butterfly_avx2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_actg_lsb_butterfly_avx2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_actg_msb_butterfly_avx2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_acgt_lsb_fold_avx2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_acgt_msb_fold_avx2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_actg_lsb_fold_avx2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_actg_msb_fold_avx2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_acgt_lsb_direct_avx2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_acgt_msb_direct_avx2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_actg_lsb_direct_avx2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_actg_msb_direct_avx2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_acgt_lsb_direct_double_avx2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_acgt_msb_direct_double_avx2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_actg_lsb_direct_double_avx2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_actg_msb_direct_double_avx2(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif

#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_validity_acgt_lsb_butterfly_avx512(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_acgt_msb_butterfly_avx512(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_actg_lsb_butterfly_avx512(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_actg_msb_butterfly_avx512(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_acgt_lsb_fold_avx512(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_acgt_msb_fold_avx512(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_actg_lsb_fold_avx512(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_actg_msb_fold_avx512(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_acgt_lsb_direct_avx512(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_acgt_msb_direct_avx512(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_actg_lsb_direct_avx512(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_actg_msb_direct_avx512(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_acgt_lsb_direct_double_avx512(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_acgt_msb_direct_double_avx512(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_actg_lsb_direct_double_avx512(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_actg_msb_direct_double_avx512(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif

#if defined(FISK_HAS_NEON)
std::uint64_t run_var_validity_acgt_lsb_butterfly_neon(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_acgt_msb_butterfly_neon(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_actg_lsb_butterfly_neon(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_actg_msb_butterfly_neon(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_acgt_lsb_fold_neon(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_acgt_msb_fold_neon(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_actg_lsb_fold_neon(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_actg_msb_fold_neon(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_acgt_lsb_direct_neon(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_acgt_msb_direct_neon(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_actg_lsb_direct_neon(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_actg_msb_direct_neon(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_acgt_lsb_direct_double_neon(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_acgt_msb_direct_double_neon(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_validity_actg_lsb_direct_double_neon(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_validity_actg_msb_direct_double_neon(
    std::string const& seq,
    fisk::PackedSequenceWithValidity<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif
