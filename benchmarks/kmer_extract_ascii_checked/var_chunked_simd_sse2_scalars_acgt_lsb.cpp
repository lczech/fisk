#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/kmer_extract/ascii_chunked_simd.hpp"
#include "chunked.hpp"
#include "kmer_extract_ascii_checked/bench.hpp"

#if defined(FISK_HAS_SSE2)

using namespace fisk;

std::uint64_t run_var_chunked_simd_sse2_scalars_acgt_lsb(
    std::string const& seq,
    std::size_t k,
    CheckedAcgtLsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_kmer_ascii_checked_chunked_simd<SimdSSE2>(
        seq, k, ChunkedWordEncoder<Encoding::kACGT, Layout::kLSB>{}, scratch,
        [&](std::size_t pos, auto kmers, auto valid) FISK_ALWAYS_INLINE_LAMBDA {
            for_each_valid_lane(pos, kmers, valid, [&](auto kmer) FISK_ALWAYS_INLINE_LAMBDA {
                sink.consume(kmer_value(kmer));
            });
        }
    );
    return sink.finalize();
}

#endif // FISK_HAS_SSE2
