#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/spaced_kmer_extract/packed_bytewise_simd.hpp"
#include "spaced_kmer_extract_packed_unchecked_single/bench.hpp"

#if defined(FISK_HAS_NEON)

using namespace fisk;

std::uint64_t run_var_bytewise_simd_block_shift_neon_vectors(
    PackedLsb const& seq,
    SpacedKmerMask<Layout::kLSB, BitExtractKernelBlockShift<SimdNEON>> const& kernel,
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

#endif // FISK_HAS_NEON
