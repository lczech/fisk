#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/spaced_kmer_extract/ascii_rolling_simd.hpp"
#include "spaced_kmer_extract_ascii_checked_multi/bench.hpp"

#if defined(FISK_HAS_SSE2)

using namespace fisk;

std::uint64_t run_var_rolling_simd_block_shift_sse2_vectors(
    std::string const& seq,
    std::vector<
        SpacedKmerMask<Layout::kMSB, BitExtractKernelBlockShift<SimdSSE2>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
) {
    return dispatch_static_span<kMaxDispatchedMasks>(kernels, [&](auto kernel_span) {
        auto sink = make_sink(sink_buffer);
        for_each_spaced_kmer_ascii_checked_rolling_simd(
            std::string_view(seq), kernel_span, CharEncoderTable<Encoding::kACGT>{},
            [&](
                std::size_t /*pos*/, std::size_t /*mask_idx*/, auto spaced_kmers, auto valid_lanes
            ) FISK_ALWAYS_INLINE_LAMBDA {
                sink.consume(kmer_value(spaced_kmers));
                do_not_optimize(valid_lanes.bits);
            }
        );
        return sink.finalize();
    });
}

#endif // FISK_HAS_SSE2
