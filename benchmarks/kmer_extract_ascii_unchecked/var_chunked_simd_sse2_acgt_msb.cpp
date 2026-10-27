#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/kmer_extract/ascii_chunked_simd.hpp"
#include "chunked.hpp"
#include "kmer_extract_ascii_unchecked/bench.hpp"

#if defined(FISK_HAS_SSE2)

using namespace fisk;

std::uint64_t run_var_chunked_simd_sse2_acgt_msb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtMsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_kmer_ascii_unchecked_chunked_simd<SimdSSE2>(
        seq, k, ChunkedWordEncoder<Encoding::kACGT, Layout::kMSB>{}, scratch,
        [&](
            std::size_t /*pos*/, auto kmers, std::size_t /*valid_count*/
        ) FISK_ALWAYS_INLINE_LAMBDA {
            sink.consume(kmer_value(kmers));
        }
    );
    return sink.finalize();
}

#endif // FISK_HAS_SSE2
