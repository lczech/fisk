#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_spaced/packed_simd.hpp"
#include "kmer_spaced_packed_single/bench.hpp"

#if defined(FISK_HAS_SSE2)

using namespace fisk;

std::uint64_t run_var_simd_butterfly_table_sse2_scalars(
    PackedLsb const& seq,
    SpacedKmerMask<Layout::kLSB, BitExtractKernelButterfly<SimdSSE2>> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_spaced_kmer_packed_simd(
        seq, kernel,
        [&](std::size_t /*pos*/, auto const& spaced_kmer) {
            sink.consume(kmer_value(spaced_kmer));
        }
    );
    return sink.finalize();
}

#endif // FISK_HAS_SSE2
