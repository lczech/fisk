#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/kmer_extract/ascii_chunked_simd.hpp"
#include "chunked.hpp"
#include "kmer_extract_ascii_checked/bench.hpp"

#if defined(FISK_HAS_SSE2)

using namespace fisk;

std::uint64_t run_var_chunked_simd_sse2_acgt_msb(
    std::string const& seq,
    std::size_t k,
    CheckedAcgtMsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_kmer_ascii_checked_chunked_simd<SimdSSE2>(
        seq, k, ChunkedWordEncoder<Encoding::kACGT, Layout::kMSB>{}, scratch,
        [&](
            std::size_t /*pos*/, auto kmers, auto valid_lanes
        ) FISK_ALWAYS_INLINE_LAMBDA {
            sink.consume(kmer_value(kmers));
            do_not_optimize(valid_lanes.bits);
        }
    );
    return sink.finalize();
}

#endif // FISK_HAS_SSE2
