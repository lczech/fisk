#include <cstddef>
#include <cstdint>

#include "fisk/kmer_extract/packed_simd.hpp"
#include "kmer_extract_packed/bench.hpp"

#if defined(FISK_HAS_AVX2)

using namespace fisk;

std::uint64_t run_var_msb_simd_wide_avx2(PackedMsb const& seq, std::size_t k)
{
    __m256i acc = _mm256_setzero_si256();
    for_each_kmer_packed_simd_wide_avx2_(seq, k, [&](__m256i v, std::size_t) noexcept {
        acc = _mm256_add_epi64(acc, v);
    });
    alignas(32) std::uint64_t buf[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(buf), acc);
    return buf[0] + buf[1] + buf[2] + buf[3];
}

#endif // FISK_HAS_AVX2
