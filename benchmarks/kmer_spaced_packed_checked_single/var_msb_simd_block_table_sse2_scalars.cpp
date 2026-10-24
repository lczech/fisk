#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_spaced/packed_simd.hpp"
#include "kmer_spaced_packed_checked_single/bench.hpp"

#if defined(FISK_HAS_SSE2)

using namespace fisk;

std::uint64_t run_var_simd_block_table_sse2_scalars(
    CheckedMsb const& seq,
    SpacedKmerMask<Layout::kMSB, BitExtractKernelBlock<SimdSSE2>> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_spaced_kmer_packed_checked_bytewise_simd(
        seq, kernel,
        [&](
            std::size_t pos, auto spaced_kmers, auto valid_lanes
        ) FISK_ALWAYS_INLINE_LAMBDA {
            for_each_valid_lane(
                pos, spaced_kmers, valid_lanes,
                [&](auto spaced_kmer) FISK_ALWAYS_INLINE_LAMBDA {
                    sink.consume(kmer_value(spaced_kmer));
                }
            );
        }
    );
    return sink.finalize();
}

#endif // FISK_HAS_SSE2
