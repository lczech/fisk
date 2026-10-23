#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/kmer_spaced/simd.hpp"
#include "fisk/core/simd_scalar.hpp"
#include "fisk/core/simd_sse2.hpp"
#include "fisk/core/simd_avx2.hpp"
#include "fisk/core/simd_avx512.hpp"
#include "fisk/core/simd_neon.hpp"
#include "sink.hpp"

/**
 * @brief Benchmark spaced k-mer extract with multiple masks at once.
 */
void bench_kmer_spaced_multi(
    std::vector<std::string> const& sequences,
    std::vector<std::vector<std::string>> const& multi_masks,
    std::ostream& csv_os
);

// Kernels compared above, each defined in its own translation unit (var_*.cpp). Each one takes,
// alongside the sequence, the piece of per-mask-set setup it needs (which carries the span k) --
// prepared once per mask set in bench.cpp and threaded in here, so that setup cost never enters
// the timed region. Guarded to match the ISA availability of the kernel each one benchmarks (see
// fisk/core/intrinsics.hpp). The ones taking spaced k-mer masks are overloaded for both Layouts,
// defined in var_msb_*.cpp and var_lsb_*.cpp, respectively. The ones taking a dispatcher create
// their sink inside of its run(), see BitExtractKernelDispatcher::run().
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

std::uint64_t run_var_bitloop(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bitloop(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_byte_table(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_byte_table(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_pext(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_pext(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

std::uint64_t run_var_block_table(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled2(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled2(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled4(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled4(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled8(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled8(
    std::string const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);

std::uint64_t run_var_butterfly_table(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractButterflyTable>
    > const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_butterfly_table(
    std::string const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractButterflyTable>
    > const& masks,
    std::vector<std::uint64_t>& sink_buffer
);

std::uint64_t run_var_simd_butterfly_table_scalar_vectors(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_scalar_scalars(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_scalar_vectors(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_scalar_scalars(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdScalar>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_scalar_vectors(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdScalar>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_scalar_scalars(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdScalar>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_scalar_vectors(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdScalar>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_scalar_scalars(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdScalar>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);

#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_simd_butterfly_table_sse2_vectors(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_sse2_scalars(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_sse2_vectors(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_sse2_scalars(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdSSE2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_sse2_vectors(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdSSE2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_sse2_scalars(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdSSE2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_sse2_vectors(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdSSE2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_sse2_scalars(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdSSE2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_simd_butterfly_table_avx2_vectors(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_avx2_scalars(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_avx2_vectors(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_avx2_scalars(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_avx2_vectors(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdAVX2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_avx2_scalars(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdAVX2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_avx2_vectors(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdAVX2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_avx2_scalars(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdAVX2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_simd_butterfly_table_avx512_vectors(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_avx512_scalars(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_avx512_vectors(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_avx512_scalars(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdAVX512>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_avx512_vectors(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdAVX512>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_avx512_scalars(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdAVX512>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_avx512_vectors(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdAVX512>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_avx512_scalars(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdAVX512>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_NEON)
std::uint64_t run_var_simd_butterfly_table_neon_vectors(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_neon_scalars(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_neon_vectors(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_butterfly_table_neon_scalars(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelButterfly<fisk::SimdNEON>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_neon_vectors(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdNEON>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_neon_scalars(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelBlock<fisk::SimdNEON>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_neon_vectors(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdNEON>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_neon_scalars(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelBlock<fisk::SimdNEON>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_simd_pext_vectors(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelPEXT>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_pext_scalars(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractKernelPEXT>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_pext_vectors(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelPEXT>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_pext_scalars(
    std::string const& seq,
    fisk::BitExtractKernelDispatcher<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractKernelPEXT>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
