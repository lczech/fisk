#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_extract/packed_simd.hpp"
#include "kmer_extract_packed/bench.hpp"

#if defined(FISK_HAS_SSE2)

using namespace fisk;

std::uint64_t run_var_lsb_simd_sse2(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_kmer_packed_simd<SimdSSE2>(
        seq, k, [&](std::size_t, auto kmers, std::size_t) noexcept {
            sink.consume(kmer_value(kmers));
        }
    );
    return sink.finalize();
}

#endif // FISK_HAS_SSE2
