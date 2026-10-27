#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/spaced_kmer_extract/ascii_chunked_simd.hpp"
#include "fisk/util/static_span.hpp"
#include "chunked.hpp"
#include "spaced_kmer_extract_ascii_unchecked_multi/bench.hpp"

#if defined(FISK_HAS_AVX512)

using namespace fisk;

std::uint64_t run_var_chunked_simd_butterfly_avx512_vectors(
    std::string const& seq,
    std::vector<ButterflyKernel<Layout::kLSB, fisk::SimdAVX512>> const& masks,
    ChunkedScratch<Layout::kLSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
) {
    constexpr Layout L = Layout::kLSB;
    return dispatch_static_span<kMaxDispatchedMasks>(masks, [&](auto mask_span) {
        auto sink = make_sink(sink_buffer);
        for_each_spaced_kmer_ascii_unchecked_chunked_simd(
            seq, mask_span, ChunkedWordEncoder<Encoding::kACGT, L>{}, scratch,
            [&](
                std::size_t /*pos*/, std::size_t /*mask_idx*/, auto spaced_kmers,
                std::size_t /*valid_count*/
            ) FISK_ALWAYS_INLINE_LAMBDA {
                sink.consume(kmer_value(spaced_kmers));
            }
        );
        return sink.finalize();
    });
}

#endif // FISK_HAS_AVX512
