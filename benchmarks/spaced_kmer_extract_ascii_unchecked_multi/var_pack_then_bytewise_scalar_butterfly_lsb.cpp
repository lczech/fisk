#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/spaced_kmer_extract/packed_bytewise_scalar.hpp"
#include "fisk/util/static_span.hpp"
#include "chunked.hpp"
#include "spaced_kmer_extract_ascii_unchecked_multi/bench.hpp"

using namespace fisk;

std::uint64_t run_var_pack_then_bytewise_scalar_butterfly(
    std::string const& seq,
    std::vector<ButterflyMask<Layout::kLSB>> const& masks,
    ChunkedScratch<Layout::kLSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
) {
    constexpr Layout L = Layout::kLSB;
    return dispatch_static_span<kMaxDispatchedMasks>(masks, [&](auto mask_span) {
        auto sink = make_sink(sink_buffer);
        pack_sequence_unchecked(seq, ChunkedWordEncoder<Encoding::kACGT, L>{}, scratch);
        for_each_spaced_kmer_packed_unchecked_bytewise_scalar(
            scratch, mask_span, bit_extract_butterfly,
            [&](std::size_t, std::size_t, auto const& spaced_kmer) FISK_ALWAYS_INLINE_LAMBDA {
                sink.consume(kmer_value(spaced_kmer));
            }
        );
        return sink.finalize();
    });
}
