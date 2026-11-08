#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/spaced_kmer_extract/packed_bytewise_simd.hpp"
#include "spaced_kmer_extract_packed_unchecked_spans/bench.hpp"

using namespace fisk;

std::uint64_t run_var_bytewise_simd_butterfly_scalar_vectors(
    PackedLsb const& seq,
    SpacedKmerMask<
        Layout::kLSB, BitExtractKernelButterfly<SimdScalar>, kSpanSweepMaxSpan
    > const& kernel,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_spaced_kmer_packed_unchecked_bytewise_simd(
        seq, kernel,
        [&](
            std::size_t /*pos*/, auto spaced_kmers, std::size_t valid_count
        ) FISK_ALWAYS_INLINE_LAMBDA {
            consume_valid_lanes(sink, spaced_kmers, valid_count);
        }
    );
    return sink.finalize();
}
