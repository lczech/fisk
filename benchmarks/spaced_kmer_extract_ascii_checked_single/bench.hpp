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
#include "sink.hpp"

/**
 * @brief Benchmark spaced k-mer extract with single masks.
 */
void bench_spaced_kmer_extract_ascii_checked_single(
    std::vector<std::string> const& sequences,
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
// for_each_valid_lane() (`_scalars`). The `_vectors` ones sum the vectors whole, valid lanes or
// not, as a caller takes them, and keep the valid lanes with a do_not_optimize() barrier, so that
// they are computed. Their sums hence differ from those of the single spaced k-mers, and they run
// in a suite.run() of their own.
std::uint64_t run_var_missh(
    std::string const& seq,
    std::size_t k,
    std::vector<std::size_t> const& naive_mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_naive(
    std::string const& seq,
    std::size_t k,
    std::vector<std::size_t> const& naive_mask,
    std::vector<std::uint64_t>& sink_buffer
);

std::uint64_t run_var_rolling_scalar_bitloop(
    std::string const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_bitloop(
    std::string const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_byte_table(
    std::string const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_byte_table(
    std::string const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_rolling_scalar_pext(
    std::string const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_pext(
    std::string const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

std::uint64_t run_var_rolling_scalar_block_table(
    std::string const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockTable> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_block_table(
    std::string const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockTable> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_block_table_unrolled2(
    std::string const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockTable> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_block_table_unrolled2(
    std::string const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockTable> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_block_table_unrolled4(
    std::string const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockTable> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_block_table_unrolled4(
    std::string const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockTable> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_block_table_unrolled8(
    std::string const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockTable> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_block_table_unrolled8(
    std::string const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockTable> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);

std::uint64_t run_var_rolling_scalar_butterfly_table(
    std::string const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractButterflyTable> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_butterfly_table(
    std::string const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractButterflyTable> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);

std::uint64_t run_var_rolling_simd_butterfly_table_scalar_vectors(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_table_scalar_scalars(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_table_scalar_vectors(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_table_scalar_scalars(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_table_scalar_vectors(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdScalar>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_table_scalar_scalars(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdScalar>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_table_scalar_vectors(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdScalar>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_table_scalar_scalars(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdScalar>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);

#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_rolling_simd_butterfly_table_sse2_vectors(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_table_sse2_scalars(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_table_sse2_vectors(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_table_sse2_scalars(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_table_sse2_vectors(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdSSE2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_table_sse2_scalars(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdSSE2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_table_sse2_vectors(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdSSE2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_table_sse2_scalars(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdSSE2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_rolling_simd_butterfly_table_avx2_vectors(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_table_avx2_scalars(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_table_avx2_vectors(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_table_avx2_scalars(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_table_avx2_vectors(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdAVX2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_table_avx2_scalars(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdAVX2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_table_avx2_vectors(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdAVX2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_table_avx2_scalars(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdAVX2>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_rolling_simd_butterfly_table_avx512_vectors(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_table_avx512_scalars(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_table_avx512_vectors(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_table_avx512_scalars(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_table_avx512_vectors(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdAVX512>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_table_avx512_scalars(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdAVX512>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_table_avx512_vectors(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdAVX512>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_table_avx512_scalars(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdAVX512>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_NEON)
std::uint64_t run_var_rolling_simd_butterfly_table_neon_vectors(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_table_neon_scalars(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_table_neon_vectors(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_table_neon_scalars(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_table_neon_vectors(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdNEON>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_table_neon_scalars(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdNEON>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_table_neon_vectors(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdNEON>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_table_neon_scalars(
    std::string const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdNEON>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_rolling_simd_pext_vectors(
    std::string const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelPEXT> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_pext_scalars(
    std::string const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelPEXT> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_pext_vectors(
    std::string const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelPEXT> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_pext_scalars(
    std::string const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelPEXT> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
