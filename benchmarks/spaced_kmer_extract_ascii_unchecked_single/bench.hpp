#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/spaced_kmer_extract/mask.hpp"
#include "chunked.hpp"
#include "sink.hpp"

/**
 * @brief Benchmark the extraction of spaced k-mers from ASCII input without checks, for single
 * masks.
 *
 * The chunked loops, see spaced_kmer_extract/ascii_chunked_scalar.hpp and
 * ascii_chunked_simd.hpp, against packing the whole sequence first and then extracting from it.
 * All of them pack with the word encoder of chunked.hpp, and hence encode invalid characters
 * (see --n-prob in main.cpp) alike, so that their sums match. The `_scalars` SIMD variants take
 * the spaced k-mers out of the vectors with for_each_valid_lane(), the `_vectors` ones sum the
 * vectors whole, whose padding lanes are zero. Each variant has a scratch of its own.
 */
void bench_spaced_kmer_extract_ascii_unchecked_single(
    std::vector<std::string> const& sequences,
    std::vector<std::string> const& masks,
    std::ostream& csv_os
);

// Kernels compared above, each defined in its own translation unit (var_*.cpp), overloaded for
// both Layouts, defined in var_*_msb.cpp and var_*_lsb.cpp, respectively.
#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_chunked_scalar_pext(
    std::string const& seq,
    PextMask<fisk::Layout::kMSB> const& mask,
    ChunkedScratch<fisk::Layout::kMSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_scalar_pext(
    std::string const& seq,
    PextMask<fisk::Layout::kLSB> const& mask,
    ChunkedScratch<fisk::Layout::kLSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
std::uint64_t run_var_chunked_scalar_butterfly(
    std::string const& seq,
    ButterflyMask<fisk::Layout::kMSB> const& mask,
    ChunkedScratch<fisk::Layout::kMSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_scalar_butterfly(
    std::string const& seq,
    ButterflyMask<fisk::Layout::kLSB> const& mask,
    ChunkedScratch<fisk::Layout::kLSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_chunked_simd_butterfly_sse2_scalars(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kMSB, fisk::SimdSSE2> const& mask,
    ChunkedScratch<fisk::Layout::kMSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_simd_butterfly_sse2_scalars(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kLSB, fisk::SimdSSE2> const& mask,
    ChunkedScratch<fisk::Layout::kLSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_chunked_simd_butterfly_avx2_scalars(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kMSB, fisk::SimdAVX2> const& mask,
    ChunkedScratch<fisk::Layout::kMSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_simd_butterfly_avx2_scalars(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kLSB, fisk::SimdAVX2> const& mask,
    ChunkedScratch<fisk::Layout::kLSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_chunked_simd_butterfly_avx512_scalars(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kMSB, fisk::SimdAVX512> const& mask,
    ChunkedScratch<fisk::Layout::kMSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_simd_butterfly_avx512_scalars(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kLSB, fisk::SimdAVX512> const& mask,
    ChunkedScratch<fisk::Layout::kLSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
#if defined(FISK_HAS_NEON)
std::uint64_t run_var_chunked_simd_butterfly_neon_scalars(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kMSB, fisk::SimdNEON> const& mask,
    ChunkedScratch<fisk::Layout::kMSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_simd_butterfly_neon_scalars(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kLSB, fisk::SimdNEON> const& mask,
    ChunkedScratch<fisk::Layout::kLSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
std::uint64_t run_var_chunked_simd_butterfly_scalar_scalars(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kMSB, fisk::SimdScalar> const& mask,
    ChunkedScratch<fisk::Layout::kMSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_simd_butterfly_scalar_scalars(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kLSB, fisk::SimdScalar> const& mask,
    ChunkedScratch<fisk::Layout::kLSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_chunked_simd_butterfly_sse2_vectors(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kMSB, fisk::SimdSSE2> const& mask,
    ChunkedScratch<fisk::Layout::kMSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_simd_butterfly_sse2_vectors(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kLSB, fisk::SimdSSE2> const& mask,
    ChunkedScratch<fisk::Layout::kLSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_chunked_simd_butterfly_avx2_vectors(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kMSB, fisk::SimdAVX2> const& mask,
    ChunkedScratch<fisk::Layout::kMSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_simd_butterfly_avx2_vectors(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kLSB, fisk::SimdAVX2> const& mask,
    ChunkedScratch<fisk::Layout::kLSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_chunked_simd_butterfly_avx512_vectors(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kMSB, fisk::SimdAVX512> const& mask,
    ChunkedScratch<fisk::Layout::kMSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_simd_butterfly_avx512_vectors(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kLSB, fisk::SimdAVX512> const& mask,
    ChunkedScratch<fisk::Layout::kLSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
#if defined(FISK_HAS_NEON)
std::uint64_t run_var_chunked_simd_butterfly_neon_vectors(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kMSB, fisk::SimdNEON> const& mask,
    ChunkedScratch<fisk::Layout::kMSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_simd_butterfly_neon_vectors(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kLSB, fisk::SimdNEON> const& mask,
    ChunkedScratch<fisk::Layout::kLSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
std::uint64_t run_var_chunked_simd_butterfly_scalar_vectors(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kMSB, fisk::SimdScalar> const& mask,
    ChunkedScratch<fisk::Layout::kMSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_simd_butterfly_scalar_vectors(
    std::string const& seq,
    ButterflyKernel<fisk::Layout::kLSB, fisk::SimdScalar> const& mask,
    ChunkedScratch<fisk::Layout::kLSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_pack_then_bytewise_scalar_butterfly(
    std::string const& seq,
    ButterflyMask<fisk::Layout::kMSB> const& mask,
    ChunkedScratch<fisk::Layout::kMSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_pack_then_bytewise_scalar_butterfly(
    std::string const& seq,
    ButterflyMask<fisk::Layout::kLSB> const& mask,
    ChunkedScratch<fisk::Layout::kLSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
