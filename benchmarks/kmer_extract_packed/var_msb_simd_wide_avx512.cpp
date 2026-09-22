#include <cstddef>
#include <cstdint>

#include "fisk/kmer_extract/packed_simd.hpp"
#include "kmer_extract_packed/bench.hpp"

#if defined(FISK_HAS_AVX512)

using namespace fisk;

std::uint64_t run_var_msb_simd_wide_avx512(PackedMsb const& seq, std::size_t k)
{
    __m512i acc = _mm512_setzero_si512();
    for_each_kmer_packed_simd_wide_avx512_(seq, k, [&](__m512i v, std::size_t) noexcept {
        acc = _mm512_add_epi64(acc, v);
    });
    alignas(64) std::uint64_t buf[8];
    _mm512_storeu_si512(buf, acc);
    return buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
}

#endif // FISK_HAS_AVX512
