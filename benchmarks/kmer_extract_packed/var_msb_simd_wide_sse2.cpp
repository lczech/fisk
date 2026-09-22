#include <cstddef>
#include <cstdint>

#include "fisk/kmer_extract/packed_simd.hpp"
#include "kmer_extract_packed/bench.hpp"

#if defined(FISK_HAS_SSE2)

using namespace fisk;

std::uint64_t run_var_msb_simd_wide_sse2(PackedMsb const& seq, std::size_t k)
{
    __m128i acc = _mm_setzero_si128();
    for_each_kmer_packed_simd_wide_sse2_(seq, k, [&](__m128i v, std::size_t) noexcept {
        acc = _mm_add_epi64(acc, v);
    });
    alignas(16) std::uint64_t buf[2];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(buf), acc);
    return buf[0] + buf[1];
}

#endif // FISK_HAS_SSE2
