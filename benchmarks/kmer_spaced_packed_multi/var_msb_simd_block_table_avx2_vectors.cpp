#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_spaced/packed_simd.hpp"
#include "kmer_spaced_packed_multi/bench.hpp"

#if defined(FISK_HAS_AVX2)

using namespace fisk;

std::uint64_t run_var_simd_block_table_avx2_vectors(
    PackedMsb const& seq,
    BitExtractKernelDispatcher<
        SpacedKmerMask<Layout::kMSB, BitExtractKernelBlock<SimdAVX2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
) {
    std::uint64_t result = 0;
    kernel.run([&](auto const& kernels_arr) {
        auto sink = make_sink(sink_buffer);
        for_each_spaced_kmer_packed_simd(
            seq, kernels_arr,
            [&](
                std::size_t /*pos*/, std::size_t /*mask_idx*/, auto spaced_kmers,
                std::size_t /*valid_count*/
            ) FISK_ALWAYS_INLINE_LAMBDA {
                sink.consume(kmer_value(spaced_kmers));
            }
        );
        result = sink.finalize();
    });
    return result;
}

#endif // FISK_HAS_AVX2
