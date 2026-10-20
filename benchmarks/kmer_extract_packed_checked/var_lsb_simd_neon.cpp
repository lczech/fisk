#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_extract/packed_simd.hpp"
#include "kmer_extract_packed_checked/bench.hpp"

#if defined(FISK_HAS_NEON)

using namespace fisk;

std::uint64_t run_var_checked_lsb_simd_neon(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_kmer_packed_checked_simd<SimdNEON>(
        seq, k, [&](std::size_t, auto kmers, KmerVectorValidLanes<SimdNEON> lanes) noexcept {
            sink.consume(valid_kmers_<SimdNEON>(kmer_value(kmers), lanes));
        }
    );
    return sink.finalize();
}

#endif // FISK_HAS_NEON
