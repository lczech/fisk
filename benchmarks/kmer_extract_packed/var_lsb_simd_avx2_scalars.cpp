#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_extract/packed_bytewise_simd.hpp"
#include "kmer_extract_packed/bench.hpp"

#if defined(FISK_HAS_AVX2)

using namespace fisk;

std::uint64_t run_var_lsb_simd_avx2_scalars(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_kmer_packed_unchecked_bytewise_simd<SimdAVX2>(
        seq, k,
        [&](std::size_t pos, auto kmers, std::size_t valid_count) FISK_ALWAYS_INLINE_LAMBDA {
            for_each_valid_lane(pos, kmers, valid_count, [&](auto kmer) FISK_ALWAYS_INLINE_LAMBDA {
                sink.consume(kmer_value(kmer));
            });
        }
    );
    return sink.finalize();
}

#endif // FISK_HAS_AVX2
