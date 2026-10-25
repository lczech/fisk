#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/spaced_kmer_extract/packed_bytewise_simd.hpp"
#include "kmer_spaced_packed_single/bench.hpp"

using namespace fisk;

std::uint64_t run_var_simd_block_table_scalar_vectors(
    PackedLsb const& seq,
    SpacedKmerMask<Layout::kLSB, BitExtractKernelBlock<SimdScalar>> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_spaced_kmer_packed_unchecked_bytewise_simd(
        seq, kernel,
        [&](
            std::size_t /*pos*/, auto spaced_kmers, std::size_t /*valid_count*/
        ) FISK_ALWAYS_INLINE_LAMBDA {
            sink.consume(kmer_value(spaced_kmers));
        }
    );
    return sink.finalize();
}
