#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/bit_extract/simd.hpp"
#include "fisk/core/types.hpp"
#include "fisk/simd/scalar.hpp"
#include "fisk/simd/sse2.hpp"
#include "fisk/simd/avx2.hpp"
#include "fisk/simd/avx512.hpp"
#include "fisk/simd/neon.hpp"
#include "sink.hpp"
#include "spaced_masks.hpp"

using CheckedMsb = fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>;
using CheckedLsb = fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>;

// The variants that hand out vectors consume only their valid lanes, see consume_valid_lanes()
// (sink.hpp), so that all variants of one case sum the same spaced k-mers and share one
// suite.run().

/**
 * @brief Benchmark spaced k-mer extraction from packed sequences with validity, emitting the valid
 * spaced k-mers, with a single mask for each span from 1 to kSpanSweepMaxSpan, see
 * spaced_mask_for_span(), each span its own case.
 */
void bench_spaced_kmer_extract_packed_checked_spans(
    std::vector<std::string> const& sequences,
    std::ostream& csv_os
);

// Kernels compared above, each defined in its own translation unit (var_*.cpp), and overloaded for
// both Layouts, defined in var_*_msb.cpp and var_*_lsb.cpp, respectively.
#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_bytewise_scalar_pext(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask, kSpanSweepMaxSpan> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_pext(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask, kSpanSweepMaxSpan> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
std::uint64_t run_var_bytewise_scalar_butterfly(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractButterflyMask, kSpanSweepMaxSpan
    > const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_butterfly(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractButterflyMask, kSpanSweepMaxSpan
    > const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_scalar_vectors(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>, kSpanSweepMaxSpan
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_scalar_vectors(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>, kSpanSweepMaxSpan
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_bytewise_simd_butterfly_sse2_vectors(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>, kSpanSweepMaxSpan
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_sse2_vectors(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>, kSpanSweepMaxSpan
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_bytewise_simd_butterfly_avx2_vectors(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>, kSpanSweepMaxSpan
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_avx2_vectors(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>, kSpanSweepMaxSpan
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_bytewise_simd_butterfly_avx512_vectors(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>, kSpanSweepMaxSpan
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_avx512_vectors(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>, kSpanSweepMaxSpan
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
#if defined(FISK_HAS_NEON)
std::uint64_t run_var_bytewise_simd_butterfly_neon_vectors(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>, kSpanSweepMaxSpan
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_butterfly_neon_vectors(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>, kSpanSweepMaxSpan
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_bytewise_simd_pext_vectors(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelPext, kSpanSweepMaxSpan
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_simd_pext_vectors(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelPext, kSpanSweepMaxSpan
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
