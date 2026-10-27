#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/spaced_kmer_extract/ascii_chunked_simd.hpp"
#include "chunked.hpp"
#include "spaced_kmer_extract_ascii_unchecked_single/bench.hpp"

#if defined(FISK_HAS_AVX512)

using namespace fisk;

std::uint64_t run_var_chunked_simd_butterfly_avx512_scalars(
    std::string const& seq,
    ButterflyKernel<Layout::kLSB, fisk::SimdAVX512> const& mask,
    ChunkedScratch<Layout::kLSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
) {
    constexpr Layout L = Layout::kLSB;
    auto sink = make_sink(sink_buffer);
    for_each_spaced_kmer_ascii_unchecked_chunked_simd(
        seq, mask, ChunkedWordEncoder<Encoding::kACGT, L>{}, scratch,
        [&](std::size_t pos, std::size_t, auto kmers, auto valid) FISK_ALWAYS_INLINE_LAMBDA {
            for_each_valid_lane(pos, kmers, valid, [&](auto const& kmer) FISK_ALWAYS_INLINE_LAMBDA {
                sink.consume(kmer_value(kmer));
            });
        }
    );
    return sink.finalize();
}

#endif // FISK_HAS_AVX512
