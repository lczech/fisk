#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/spaced_kmer_extract/mask.hpp"
#include "chunked.hpp"
#include "sink.hpp"
#include "spaced_masks.hpp"

// The variants that hand out vectors consume only their valid lanes, see consume_valid_lanes()
// (sink.hpp), so that all variants of one case sum the same spaced k-mers and share one
// suite.run().

/**
 * @brief Benchmark the chunked extraction of the valid spaced k-mers from ASCII input, with a
 * single mask for each span from 1 to kSpanSweepMaxSpan, see spaced_mask_for_span(), each span its
 * own case.
 *
 * The chunked loops, see spaced_kmer_extract/ascii_chunked_scalar.hpp and ascii_chunked_simd.hpp,
 * with the word encoder of chunked.hpp. Each variant has a scratch of its own.
 */
void bench_spaced_kmer_extract_ascii_checked_spans(
    std::vector<std::string> const& sequences,
    std::ostream& csv_os
);

// Kernels compared above, each defined in its own translation unit (var_*.cpp), and overloaded for
// both Layouts, defined in var_*_msb.cpp and var_*_lsb.cpp, respectively.
#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_chunked_scalar_pext(
    std::string const& seq,
    PextMask<fisk::Layout::kMSB, kSpanSweepMaxSpan> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kMSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_scalar_pext(
    std::string const& seq,
    PextMask<fisk::Layout::kLSB, kSpanSweepMaxSpan> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kLSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
std::uint64_t run_var_chunked_scalar_butterfly(
    std::string const& seq,
    ButterflyMask<fisk::Layout::kMSB, kSpanSweepMaxSpan> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kMSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_scalar_butterfly(
    std::string const& seq,
    ButterflyMask<fisk::Layout::kLSB, kSpanSweepMaxSpan> const& mask,
    ChunkedCheckedScratch<fisk::Layout::kLSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_simd_butterfly_scalar_vectors(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kMSB, fisk::SimdScalar, kSpanSweepMaxSpan> const& kernel,
    ChunkedCheckedScratch<fisk::Layout::kMSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_simd_butterfly_scalar_vectors(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kLSB, fisk::SimdScalar, kSpanSweepMaxSpan> const& kernel,
    ChunkedCheckedScratch<fisk::Layout::kLSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_chunked_simd_butterfly_sse2_vectors(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kMSB, fisk::SimdSSE2, kSpanSweepMaxSpan> const& kernel,
    ChunkedCheckedScratch<fisk::Layout::kMSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_simd_butterfly_sse2_vectors(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kLSB, fisk::SimdSSE2, kSpanSweepMaxSpan> const& kernel,
    ChunkedCheckedScratch<fisk::Layout::kLSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_chunked_simd_butterfly_avx2_vectors(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kMSB, fisk::SimdAVX2, kSpanSweepMaxSpan> const& kernel,
    ChunkedCheckedScratch<fisk::Layout::kMSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_simd_butterfly_avx2_vectors(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kLSB, fisk::SimdAVX2, kSpanSweepMaxSpan> const& kernel,
    ChunkedCheckedScratch<fisk::Layout::kLSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_chunked_simd_butterfly_avx512_vectors(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kMSB, fisk::SimdAVX512, kSpanSweepMaxSpan> const& kernel,
    ChunkedCheckedScratch<fisk::Layout::kMSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_simd_butterfly_avx512_vectors(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kLSB, fisk::SimdAVX512, kSpanSweepMaxSpan> const& kernel,
    ChunkedCheckedScratch<fisk::Layout::kLSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
#if defined(FISK_HAS_NEON)
std::uint64_t run_var_chunked_simd_butterfly_neon_vectors(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kMSB, fisk::SimdNEON, kSpanSweepMaxSpan> const& kernel,
    ChunkedCheckedScratch<fisk::Layout::kMSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_simd_butterfly_neon_vectors(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kLSB, fisk::SimdNEON, kSpanSweepMaxSpan> const& kernel,
    ChunkedCheckedScratch<fisk::Layout::kLSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
