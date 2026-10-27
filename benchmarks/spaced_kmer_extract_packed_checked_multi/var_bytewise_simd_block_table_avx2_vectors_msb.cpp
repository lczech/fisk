#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/spaced_kmer_extract/packed_bytewise_simd.hpp"
#include "spaced_kmer_extract_packed_checked_multi/bench.hpp"

#if defined(FISK_HAS_AVX2)

using namespace fisk;

std::uint64_t run_var_bytewise_simd_block_table_avx2_vectors(
    CheckedMsb const& seq,
    std::vector<
        SpacedKmerMask<Layout::kMSB, BitExtractKernelBlock<SimdAVX2>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
) {
    return dispatch_static_span<kMaxDispatchedMasks>(kernels, [&](auto kernel_span) {
        auto sink = make_sink(sink_buffer);
        for_each_spaced_kmer_packed_checked_bytewise_simd(
            seq, kernel_span,
            [&](
                std::size_t /*pos*/, std::size_t /*mask_idx*/, auto spaced_kmers,
                auto valid_lanes
            ) FISK_ALWAYS_INLINE_LAMBDA {
                sink.consume(kmer_value(spaced_kmers));
                do_not_optimize(valid_lanes.bits);
            }
        );
        return sink.finalize();
    });
}

#endif // FISK_HAS_AVX2
