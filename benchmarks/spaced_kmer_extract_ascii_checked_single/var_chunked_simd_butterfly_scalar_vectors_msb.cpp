#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/spaced_kmer_extract/ascii_chunked_simd.hpp"
#include "chunked.hpp"
#include "spaced_kmer_extract_ascii_checked_single/bench.hpp"

using namespace fisk;

std::uint64_t run_var_chunked_simd_butterfly_scalar_vectors(
    std::string const& seq,
    ButterflyKernel<Layout::kMSB, fisk::SimdScalar> const& mask,
    ChunkedCheckedScratch<Layout::kMSB>& scratch,
    std::vector<std::uint64_t>& sink_buffer
) {
    constexpr Layout L = Layout::kMSB;
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
