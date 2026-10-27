#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/spaced_kmer_extract/ascii_chunked_scalar.hpp"
#include "chunked.hpp"
#include "spaced_kmer_extract_ascii_unchecked_single/bench.hpp"

#if defined(FISK_HAS_BMI2)

using namespace fisk;

std::uint64_t run_var_chunked_scalar_pext(
    std::string const& seq,
    PextMask<Layout::kMSB> const& mask,
    ChunkedScratch<Layout::kMSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
) {
    constexpr Layout L = Layout::kMSB;
    auto sink = make_sink(sink_buffer);
    for_each_spaced_kmer_ascii_unchecked_chunked_scalar(
        seq, mask, ChunkedWordEncoder<Encoding::kACGT, L>{}, bit_extract_pext, scratch,
        [&](std::size_t, std::size_t, auto const& spaced_kmer) FISK_ALWAYS_INLINE_LAMBDA {
            sink.consume(kmer_value(spaced_kmer));
        }
    );
    return sink.finalize();
}

#endif // FISK_HAS_BMI2
