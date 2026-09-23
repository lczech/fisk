#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/kmer_spaced/simd.hpp"
#include "sink.hpp"

/**
 * @brief Benchmark spaced k-mer extract with single masks.
 */
void bench_kmer_spaced_single(
    std::vector<std::string> const& sequences,
    std::vector<std::string> const& masks,
    std::ostream& csv_os
);

// Kernels compared above, each defined in its own translation unit (var_*.cpp). Each one takes,
// alongside the sequence and k, the piece of per-mask setup it needs -- prepared once per mask in
// bench.cpp and threaded in here, so that setup cost never enters the timed region. Guarded to
// match the ISA availability of the kernel each one benchmarks (see fisk/core/intrinsics.hpp).
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

std::uint64_t run_var_bitloop(
    std::string const& seq,
    std::size_t k,
    fisk::BitExtractMask const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_byte_table(
    std::string const& seq,
    std::size_t k,
    fisk::BitExtractMask const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_pext(
    std::string const& seq,
    std::size_t k,
    fisk::BitExtractMask const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

std::uint64_t run_var_block_table(
    std::string const& seq,
    std::size_t k,
    fisk::BitExtractBlockTable const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled2(
    std::string const& seq,
    std::size_t k,
    fisk::BitExtractBlockTable const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled4(
    std::string const& seq,
    std::size_t k,
    fisk::BitExtractBlockTable const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled8(
    std::string const& seq,
    std::size_t k,
    fisk::BitExtractBlockTable const& mask,
    std::vector<std::uint64_t>& sink_buffer
);

std::uint64_t run_var_butterfly_table(
    std::string const& seq,
    std::size_t k,
    fisk::BitExtractButterflyTable const& mask,
    std::vector<std::uint64_t>& sink_buffer
);

std::uint64_t run_var_simd_butterfly_table_scalar(
    std::string const& seq,
    std::size_t k,
    fisk::BitExtractKernelButterflyScalar const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_scalar(
    std::string const& seq,
    std::size_t k,
    fisk::BitExtractKernelBlockScalar<> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);

#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_simd_butterfly_table_sse2(
    std::string const& seq,
    std::size_t k,
    fisk::BitExtractKernelButterflySSE2 const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_sse2(
    std::string const& seq,
    std::size_t k,
    fisk::BitExtractKernelBlockSSE2<> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_simd_butterfly_table_avx2(
    std::string const& seq,
    std::size_t k,
    fisk::BitExtractKernelButterflyAVX2 const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_avx2(
    std::string const& seq,
    std::size_t k,
    fisk::BitExtractKernelBlockAVX2<> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_simd_butterfly_table_avx512(
    std::string const& seq,
    std::size_t k,
    fisk::BitExtractKernelButterflyAVX512 const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_avx512(
    std::string const& seq,
    std::size_t k,
    fisk::BitExtractKernelBlockAVX512<> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_NEON)
std::uint64_t run_var_simd_butterfly_table_neon(
    std::string const& seq,
    std::size_t k,
    fisk::BitExtractKernelButterflyNEON const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_simd_block_table_neon(
    std::string const& seq,
    std::size_t k,
    fisk::BitExtractKernelBlockNEON<> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_simd_pext(
    std::string const& seq,
    std::size_t k,
    fisk::BitExtractKernelPEXT<> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
