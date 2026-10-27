#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/spaced_kmer_extract/ascii_chunked_scalar.hpp"
#include "fisk/util/static_span.hpp"
#include "chunked.hpp"
#include "spaced_kmer_extract_ascii_unchecked_multi/bench.hpp"

#if defined(FISK_HAS_BMI2)

using namespace fisk;

std::uint64_t run_var_chunked_scalar_pext(
    std::string const& seq,
    std::vector<PextMask<Layout::kLSB>> const& masks,
    ChunkedScratch<Layout::kLSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
) {
    constexpr Layout L = Layout::kLSB;
    return dispatch_static_span<kMaxDispatchedMasks>(masks, [&](auto mask_span) {
        auto sink = make_sink(sink_buffer);
        for_each_spaced_kmer_ascii_unchecked_chunked_scalar(
            seq, mask_span, ChunkedWordEncoder<Encoding::kACGT, L>{}, bit_extract_pext, scratch,
            [&](std::size_t, std::size_t, auto const& spaced_kmer) FISK_ALWAYS_INLINE_LAMBDA {
                sink.consume(kmer_value(spaced_kmer));
            }
        );
        return sink.finalize();
    });
}

#endif // FISK_HAS_BMI2
