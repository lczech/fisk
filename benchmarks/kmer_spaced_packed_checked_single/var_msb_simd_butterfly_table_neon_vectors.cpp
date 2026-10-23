#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_spaced/packed_simd.hpp"
#include "kmer_spaced_packed_checked_single/bench.hpp"

#if defined(FISK_HAS_NEON)

using namespace fisk;

std::uint64_t run_var_simd_butterfly_table_neon_vectors(
    CheckedMsb const& seq,
    SpacedKmerMask<Layout::kMSB, BitExtractKernelButterfly<SimdNEON>> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_spaced_kmer_packed_checked_simd(
        seq, kernel,
        [&](
            std::size_t /*pos*/, auto spaced_kmers, auto valid_lanes
        ) FISK_ALWAYS_INLINE_LAMBDA {
            sink.consume(kmer_value(spaced_kmers));
            do_not_optimize(valid_lanes.bits);
        }
    );
    return sink.finalize();
}

#endif // FISK_HAS_NEON
