#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/spaced_kmer_extract/ascii_chunked_simd.hpp"
#include "chunked.hpp"
#include "spaced_kmer_extract_ascii_checked_single/bench.hpp"

#if defined(FISK_HAS_AVX2)

using namespace fisk;

std::uint64_t run_var_chunked_simd_butterfly_avx2_vectors(
    std::string const& seq,
    ButterflyKernel<Layout::kLSB, fisk::SimdAVX2> const& mask,
    ChunkedCheckedScratch<Layout::kLSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
) {
    constexpr Layout L = Layout::kLSB;
    auto sink = make_sink(sink_buffer);
    for_each_spaced_kmer_ascii_checked_chunked_simd(
        seq, mask, ChunkedWordEncoder<Encoding::kACGT, L>{}, scratch,
        [&](
            std::size_t /*pos*/, auto spaced_kmers, auto valid_lanes
        ) FISK_ALWAYS_INLINE_LAMBDA {
            sink.consume(kmer_value(spaced_kmers));
            do_not_optimize(valid_lanes.bits);
        }
    );
    return sink.finalize();
}

#endif // FISK_HAS_AVX2
