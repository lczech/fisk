#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/kmer_spaced/simd.hpp"

/**
 * @brief Benchmark spaced k-mer extract with multiple masks at once.
 */
void bench_kmer_spaced_multi(
    std::vector<std::string> const& sequences,
    std::vector<std::vector<std::string>> const& multi_masks,
    std::ostream& csv_os
);

// Kernels compared above, each defined in its own translation unit (var_*.cpp). Each one takes,
// alongside the sequence and k, the piece of per-mask-set setup it needs -- prepared once per mask
// set in bench.cpp and threaded in here, so that setup cost never enters the timed region. Guarded
// to match the ISA availability of the kernel each one benchmarks (see fisk/core/intrinsics.hpp).
std::uint64_t run_var_naive(std::string const& seq, std::size_t k, std::vector<std::vector<std::size_t>> const& naive_masks);

std::uint64_t run_var_bitloop(std::string const& seq, std::size_t k, std::vector<fisk::BitExtractMask> const& masks);
std::uint64_t run_var_byte_table(std::string const& seq, std::size_t k, std::vector<fisk::BitExtractMask> const& masks);
#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_pext(std::string const& seq, std::size_t k, std::vector<fisk::BitExtractMask> const& masks);
#endif

std::uint64_t run_var_block_table(std::string const& seq, std::size_t k, std::vector<fisk::BitExtractBlockTable> const& masks);
std::uint64_t run_var_block_table_unrolled2(std::string const& seq, std::size_t k, std::vector<fisk::BitExtractBlockTable> const& masks);
std::uint64_t run_var_block_table_unrolled4(std::string const& seq, std::size_t k, std::vector<fisk::BitExtractBlockTable> const& masks);
std::uint64_t run_var_block_table_unrolled8(std::string const& seq, std::size_t k, std::vector<fisk::BitExtractBlockTable> const& masks);

std::uint64_t run_var_butterfly_table(std::string const& seq, std::size_t k, std::vector<fisk::BitExtractButterflyTable> const& masks);

std::uint64_t run_var_simd_butterfly_table_scalar(std::string const& seq, std::size_t k, fisk::BitExtractKernelDispatcher<fisk::BitExtractKernelButterflyScalar> const& kernel);
std::uint64_t run_var_simd_block_table_scalar(std::string const& seq, std::size_t k, fisk::BitExtractKernelDispatcher<fisk::BitExtractKernelBlockScalar<>> const& kernel);

#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_simd_butterfly_table_sse2(std::string const& seq, std::size_t k, fisk::BitExtractKernelDispatcher<fisk::BitExtractKernelButterflySSE2> const& kernel);
std::uint64_t run_var_simd_block_table_sse2(std::string const& seq, std::size_t k, fisk::BitExtractKernelDispatcher<fisk::BitExtractKernelBlockSSE2<>> const& kernel);
#endif

#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_simd_butterfly_table_avx2(std::string const& seq, std::size_t k, fisk::BitExtractKernelDispatcher<fisk::BitExtractKernelButterflyAVX2> const& kernel);
std::uint64_t run_var_simd_block_table_avx2(std::string const& seq, std::size_t k, fisk::BitExtractKernelDispatcher<fisk::BitExtractKernelBlockAVX2<>> const& kernel);
#endif

#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_simd_butterfly_table_avx512(std::string const& seq, std::size_t k, fisk::BitExtractKernelDispatcher<fisk::BitExtractKernelButterflyAVX512> const& kernel);
std::uint64_t run_var_simd_block_table_avx512(std::string const& seq, std::size_t k, fisk::BitExtractKernelDispatcher<fisk::BitExtractKernelBlockAVX512<>> const& kernel);
#endif

#if defined(FISK_HAS_NEON)
std::uint64_t run_var_simd_butterfly_table_neon(std::string const& seq, std::size_t k, fisk::BitExtractKernelDispatcher<fisk::BitExtractKernelButterflyNEON> const& kernel);
std::uint64_t run_var_simd_block_table_neon(std::string const& seq, std::size_t k, fisk::BitExtractKernelDispatcher<fisk::BitExtractKernelBlockNEON<>> const& kernel);
#endif

#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_simd_pext(std::string const& seq, std::size_t k, fisk::BitExtractKernelDispatcher<fisk::BitExtractKernelPEXT<>> const& kernel);
#endif
