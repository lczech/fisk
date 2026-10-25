#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_extract/packed_bytewise_simd.hpp"
#include "kmer_extract_packed_checked/bench.hpp"

#if defined(FISK_HAS_SSE2)

using namespace fisk;

std::uint64_t run_var_checked_msb_simd_sse2_scalars(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_kmer_packed_checked_bytewise_simd<SimdSSE2>(
        seq, k, [&](std::size_t pos, auto kmers, auto valid_lanes) FISK_ALWAYS_INLINE_LAMBDA {
            for_each_valid_lane(pos, kmers, valid_lanes, [&](auto kmer) FISK_ALWAYS_INLINE_LAMBDA {
                sink.consume(kmer_value(kmer));
            });
        }
    );
    return sink.finalize();
}

#endif // FISK_HAS_SSE2
