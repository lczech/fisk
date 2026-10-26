#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_extract/packed_bytewise_simd.hpp"
#include "kmer_extract_packed_unchecked/bench.hpp"

#if defined(FISK_HAS_AVX512)

using namespace fisk;

std::uint64_t run_var_unchecked_bytewise_simd_narrow_width_avx512_lsb(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_kmer_packed_bytewise_simd_narrow_width_<SimdAVX512>(
        seq, k,
        [&](std::size_t, __m512i v, std::size_t) FISK_ALWAYS_INLINE_LAMBDA {
            sink.consume(v);
        }
    );
    return sink.finalize();
}

#endif // FISK_HAS_AVX512
