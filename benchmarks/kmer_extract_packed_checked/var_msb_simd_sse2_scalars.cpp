#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_extract/packed_simd.hpp"
#include "kmer_extract_packed_checked/bench.hpp"

#if defined(FISK_HAS_SSE2)

using namespace fisk;

std::uint64_t run_var_checked_msb_simd_sse2_scalars(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_kmer_packed_checked_simd<SimdSSE2>(
        seq, k, [&](std::size_t pos, auto kmers, auto valid_lanes) noexcept {
            for_each_valid_lane(pos, kmers, valid_lanes, [&](auto kmer) noexcept {
                sink.consume(kmer_value(kmer));
            });
        }
    );
    return sink.finalize();
}

#endif // FISK_HAS_SSE2
