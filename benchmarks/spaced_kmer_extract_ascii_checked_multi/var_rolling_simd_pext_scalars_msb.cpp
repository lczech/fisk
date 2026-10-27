#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/spaced_kmer_extract/ascii_rolling_simd.hpp"
#include "spaced_kmer_extract_ascii_checked_multi/bench.hpp"

#if defined(FISK_HAS_BMI2)

using namespace fisk;

std::uint64_t run_var_rolling_simd_pext_scalars(
    std::string const& seq,
    std::vector<SpacedKmerMask<Layout::kMSB, BitExtractKernelPext>> const& kernels,
    std::vector<std::uint64_t>& sink_buffer
) {
    return dispatch_static_span<kMaxDispatchedMasks>(kernels, [&](auto kernel_span) {
        auto sink = make_sink(sink_buffer);
        for_each_spaced_kmer_ascii_checked_rolling_simd(
            std::string_view(seq), kernel_span, CharEncoderTable<Encoding::kACGT>{},
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
        return sink.finalize();
    });
}

#endif // FISK_HAS_BMI2
