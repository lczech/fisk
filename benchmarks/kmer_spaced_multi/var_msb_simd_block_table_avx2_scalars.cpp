#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/kmer_spaced/simd.hpp"
#include "kmer_spaced_multi/bench.hpp"

#if defined(FISK_HAS_AVX2)

using namespace fisk;

std::uint64_t run_var_simd_block_table_avx2_scalars(
    std::string const& seq,
    BitExtractKernelDispatcher<
        SpacedKmerMask<Layout::kMSB, BitExtractKernelBlock<SimdAVX2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
) {
    std::uint64_t result = 0;
    kernel.run([&](auto const& kernels_arr) {
        auto sink = make_sink(sink_buffer);
        for_each_spaced_kmer_ascii_checked_rolling_simd(
            std::string_view(seq), kernels_arr, CharEncoderTable<Encoding::kACGT>{},
            [&](
                std::size_t pos, std::size_t /*mask_idx*/, auto spaced_kmers, auto valid_lanes
            ) FISK_ALWAYS_INLINE_LAMBDA {
                for_each_valid_lane(
                    pos, spaced_kmers, valid_lanes,
                    [&](auto spaced_kmer) FISK_ALWAYS_INLINE_LAMBDA {
                        sink.consume(kmer_value(spaced_kmer));
                    }
                );
            }
        );
        result = sink.finalize();
    });
    return result;
}

#endif // FISK_HAS_AVX2
