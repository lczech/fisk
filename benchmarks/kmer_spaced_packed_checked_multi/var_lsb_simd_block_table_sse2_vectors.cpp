#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_spaced/packed_simd.hpp"
#include "kmer_spaced_packed_checked_multi/bench.hpp"

#if defined(FISK_HAS_SSE2)

using namespace fisk;

std::uint64_t run_var_simd_block_table_sse2_vectors(
    CheckedLsb const& seq,
    BitExtractKernelDispatcher<
        SpacedKmerMask<Layout::kLSB, BitExtractKernelBlock<SimdSSE2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
) {
    std::uint64_t result = 0;
    kernel.run([&](auto const& kernels_arr) {
        auto sink = make_sink(sink_buffer);
        for_each_spaced_kmer_packed_checked_bytewise_simd(
            seq, kernels_arr,
            [&](
                std::size_t /*pos*/, std::size_t /*mask_idx*/, auto spaced_kmers,
                auto valid_lanes
            ) FISK_ALWAYS_INLINE_LAMBDA {
                sink.consume(kmer_value(spaced_kmers));
                do_not_optimize(valid_lanes.bits);
            }
        );
        result = sink.finalize();
    });
    return result;
}

#endif // FISK_HAS_SSE2
