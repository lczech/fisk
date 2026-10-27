#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/spaced_kmer_extract/ascii_chunked_simd.hpp"
#include "chunked.hpp"
#include "spaced_kmer_extract_ascii_unchecked_single/bench.hpp"

#if defined(FISK_HAS_NEON)

using namespace fisk;

std::uint64_t run_var_chunked_simd_butterfly_neon_vectors(
    std::string const& seq,
    ButterflyKernel<Layout::kMSB, fisk::SimdNEON> const& mask,
    ChunkedScratch<Layout::kMSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
) {
    constexpr Layout L = Layout::kMSB;
    auto sink = make_sink(sink_buffer);
    for_each_spaced_kmer_ascii_unchecked_chunked_simd(
        seq, mask, ChunkedWordEncoder<Encoding::kACGT, L>{}, scratch,
        [&](
            std::size_t /*pos*/, auto spaced_kmers, std::size_t /*valid_count*/
        ) FISK_ALWAYS_INLINE_LAMBDA {
            sink.consume(kmer_value(spaced_kmers));
        }
    );
    return sink.finalize();
}

#endif // FISK_HAS_NEON
