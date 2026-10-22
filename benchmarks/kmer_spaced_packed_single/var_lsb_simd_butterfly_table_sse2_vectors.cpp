#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_spaced/packed_simd.hpp"
#include "kmer_spaced_packed_single/bench.hpp"

#if defined(FISK_HAS_SSE2)

using namespace fisk;

std::uint64_t run_var_simd_butterfly_table_sse2_vectors(
    PackedLsb const& seq,
    SpacedKmerMask<Layout::kLSB, BitExtractKernelButterfly<SimdSSE2>> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_spaced_kmer_packed_simd(
        seq, kernel,
        [&](
            std::size_t /*pos*/, std::size_t /*mask_idx*/, __m128i spaced_kmers,
            std::size_t /*valid_count*/
        ) FISK_ALWAYS_INLINE_LAMBDA {
            sink.consume(spaced_kmers);
        }
    );
    return sink.finalize();
}

#endif // FISK_HAS_SSE2
