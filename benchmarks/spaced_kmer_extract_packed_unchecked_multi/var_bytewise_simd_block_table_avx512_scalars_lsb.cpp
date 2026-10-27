#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/spaced_kmer_extract/packed_bytewise_simd.hpp"
#include "spaced_kmer_extract_packed_unchecked_multi/bench.hpp"

#if defined(FISK_HAS_AVX512)

using namespace fisk;

std::uint64_t run_var_bytewise_simd_block_table_avx512_scalars(
    PackedLsb const& seq,
    std::vector<
        SpacedKmerMask<Layout::kLSB, BitExtractKernelBlock<SimdAVX512>>
    > const& kernels,
    std::vector<std::uint64_t>& sink_buffer
) {
    return dispatch_static_span<kMaxDispatchedMasks>(kernels, [&](auto kernel_span) {
        auto sink = make_sink(sink_buffer);
        for_each_spaced_kmer_packed_unchecked_bytewise_simd(
            seq, kernel_span,
            [&](
                std::size_t pos, std::size_t /*mask_idx*/, auto spaced_kmers,
                std::size_t valid_count
            ) FISK_ALWAYS_INLINE_LAMBDA {
                for_each_valid_lane(
                    pos, spaced_kmers, valid_count,
                    [&](auto spaced_kmer) FISK_ALWAYS_INLINE_LAMBDA {
                        sink.consume(kmer_value(spaced_kmer));
                    }
                );
            }
        );
        return sink.finalize();
    });
}

#endif // FISK_HAS_AVX512
