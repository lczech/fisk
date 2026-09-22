#include <cstddef>
#include <cstdint>

#include "fisk/kmer_extract/packed_simd.hpp"
#include "kmer_extract_packed/bench.hpp"

#if defined(FISK_HAS_NEON)

using namespace fisk;

std::uint64_t run_var_lsb_simd_wide_neon(PackedLsb const& seq, std::size_t k)
{
    uint64x2_t acc = vdupq_n_u64(0);
    for_each_kmer_packed_simd_wide_neon_(seq, k, [&](uint64x2_t v, std::size_t) noexcept {
        acc = vaddq_u64(acc, v);
    });
    return vgetq_lane_u64(acc, 0) + vgetq_lane_u64(acc, 1);
}

#endif // FISK_HAS_NEON
