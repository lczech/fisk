#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/spaced_kmer_extract/packed_bytewise_scalar.hpp"
#include "spaced_kmer_extract_packed_unchecked_multi/bench.hpp"

using namespace fisk;

std::uint64_t run_var_bytewise_scalar_butterfly(
    PackedLsb const& seq,
    std::vector<SpacedKmerMask<Layout::kLSB, BitExtractButterflyMask>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
) {
    return dispatch_static_span<kMaxDispatchedMasks>(masks, [&](auto mask_span) {
        auto sink = make_sink(sink_buffer);
        for_each_spaced_kmer_packed_unchecked_bytewise_scalar(
            seq,
            mask_span,
            bit_extract_butterfly,
            [&](
                std::size_t /*pos*/, std::size_t /*mask_idx*/, auto const& spaced_kmer
            ) FISK_ALWAYS_INLINE_LAMBDA {
                sink.consume(kmer_value(spaced_kmer));
            }
        );
        return sink.finalize();
    });
}
