#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_extract/packed_simd.hpp"
#include "kmer_extract_packed_checked/bench.hpp"

#if defined(FISK_HAS_SSE2)

using namespace fisk;

std::uint64_t run_var_checked_lsb_simd_full_width_sse2(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    auto const consume = [&](
        std::size_t, __m128i v, SimdVectorValidLanes<SimdSSE2> lanes
    ) FISK_ALWAYS_INLINE_LAMBDA {
        sink.consume(valid_kmers_<SimdSSE2>(v, lanes));
    };
    for_each_kmer_packed_simd_full_width_<SimdSSE2>(seq, k, consume);
    return sink.finalize();
}

#endif // FISK_HAS_SSE2
