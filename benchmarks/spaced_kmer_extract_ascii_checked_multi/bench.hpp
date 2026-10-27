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
 * @brief Benchmark spaced k-mer extract with multiple masks at once.
 */
void bench_spaced_kmer_extract_ascii_checked_multi(
    std::vector<std::string> const& sequences,
    std::vector<std::vector<std::string>> const& multi_masks,
    std::ostream& csv_os
);

// Kernels compared above, each defined in its own translation unit (var_*.cpp). Each one takes,
// alongside the sequence, the piece of per-mask-set setup it needs (which carries the span k) --
// prepared once per mask set in bench.cpp and threaded in here, so that setup cost never enters
// the timed region. Guarded to match the ISA availability of the kernel each one benchmarks (see
// fisk/util/intrinsics.hpp). The ones taking spaced k-mer masks are overloaded for both Layouts,
// defined in var_*_msb.cpp and var_*_lsb.cpp, respectively. They hand their masks or kernels on
// as a span of fixed size, see dispatch_static_span(), and create their sink inside of it.
//
// The SIMD ones take their spaced k-mers either as vectors (`_vectors`), or one at a time with
// for_each_valid_lane() (`_scalars`). The `_vectors` ones sum the vectors whole, valid lanes or
// not, as a caller takes them, and keep the valid lanes with a do_not_optimize() barrier, so that
// they are computed. Their sums hence differ from those of the single spaced k-mers, and they run
// in a suite.run() of their own.
std::uint64_t run_var_naive(
    std::string const& seq,
    std::size_t k,
    std::vector<std::vector<std::size_t>> const& naive_masks,
    std::vector<std::uint64_t>& sink_buffer
);

std::uint64_t run_var_rolling_scalar_bit_loop(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_bit_loop(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_byte_lookup(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_byte_lookup(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_rolling_scalar_pext(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_pext(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

std::uint64_t run_var_rolling_scalar_block_shift(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_block_shift(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_block_shift_unrolled2(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_block_shift_unrolled2(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_block_shift_unrolled4(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_block_shift_unrolled4(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_block_shift_unrolled8(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_block_shift_unrolled8(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);

std::uint64_t run_var_rolling_scalar_butterfly(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractButterflyTable>
    > const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_scalar_butterfly(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractButterflyTable>
    > const& masks,
    std::vector<std::uint64_t>& sink_buffer
);

std::uint64_t run_var_rolling_simd_butterfly_scalar_vectors(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_scalar_scalars(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_scalar_vectors(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_scalar_scalars(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_scalar_vectors(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdScalar>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_scalar_scalars(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdScalar>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_scalar_vectors(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdScalar>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_scalar_scalars(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdScalar>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);

#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_rolling_simd_butterfly_sse2_vectors(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_sse2_scalars(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_sse2_vectors(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_sse2_scalars(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_sse2_vectors(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdSSE2>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_sse2_scalars(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdSSE2>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_sse2_vectors(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdSSE2>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_sse2_scalars(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdSSE2>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_rolling_simd_butterfly_avx2_vectors(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_avx2_scalars(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_avx2_vectors(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_avx2_scalars(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_avx2_vectors(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX2>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_avx2_scalars(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX2>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_avx2_vectors(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX2>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_avx2_scalars(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX2>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_rolling_simd_butterfly_avx512_vectors(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_avx512_scalars(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_avx512_vectors(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_avx512_scalars(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_avx512_vectors(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX512>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_avx512_scalars(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX512>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_avx512_vectors(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX512>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_avx512_scalars(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdAVX512>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_NEON)
std::uint64_t run_var_rolling_simd_butterfly_neon_vectors(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_neon_scalars(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_neon_vectors(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_butterfly_neon_scalars(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_neon_vectors(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdNEON>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_neon_scalars(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlockShift<fisk::SimdNEON>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_neon_vectors(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdNEON>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_block_shift_neon_scalars(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlockShift<fisk::SimdNEON>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_rolling_simd_pext_vectors(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelPEXT>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_pext_scalars(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelPEXT>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_pext_vectors(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelPEXT>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_rolling_simd_pext_scalars(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelPEXT>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
