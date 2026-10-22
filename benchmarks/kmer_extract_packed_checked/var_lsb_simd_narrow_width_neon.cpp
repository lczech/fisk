#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_extract/packed_simd.hpp"
#include "kmer_extract_packed_checked/bench.hpp"

#if defined(FISK_HAS_NEON)

using namespace fisk;

std::uint64_t run_var_checked_lsb_simd_narrow_width_neon(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    auto const consume = [&](
        std::size_t, uint64x2_t v, KmerVectorValidLanes<SimdNEON> lanes
    ) FISK_ALWAYS_INLINE_LAMBDA {
        sink.consume(valid_kmers_<SimdNEON>(v, lanes));
    };
    for_each_kmer_packed_simd_narrow_width_<SimdNEON>(seq, k, consume);
    return sink.finalize();
}

#endif // FISK_HAS_NEON
