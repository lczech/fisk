#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/kmer_spaced/simd.hpp"
#include "kmer_spaced_multi/bench.hpp"

#if defined(FISK_HAS_AVX2)

using namespace fisk;

std::uint64_t run_var_simd_block_table_avx2_by_mask(
    std::string const& seq,
    BitExtractKernelDispatcher<
        SpacedKmerMask<Layout::kLSB, BitExtractKernelBlock<SimdAVX2>>
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
) {
    std::uint64_t result = 0;
    kernel.run([&](auto const& kernels_arr) {
        auto sink = make_sink(sink_buffer);
        for_each_spaced_kmer_simd_by_mask(
            std::string_view(seq), kernels_arr, CharEncoderTable<Encoding::kACGT>{},
            [&](
                std::size_t /*pos*/, std::size_t /*mask_idx*/, auto const& wmer
            ) FISK_ALWAYS_INLINE_LAMBDA {
                sink.consume(kmer_value(wmer));
            }
        );
        result = sink.finalize();
    });
    return result;
}

#endif // FISK_HAS_AVX2
