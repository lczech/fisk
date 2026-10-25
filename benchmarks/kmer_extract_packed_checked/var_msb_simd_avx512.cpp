#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_extract/packed_bytewise_simd.hpp"
#include "kmer_extract_packed_checked/bench.hpp"

#if defined(FISK_HAS_AVX512)

using namespace fisk;

std::uint64_t run_var_checked_msb_simd_avx512(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_kmer_packed_checked_bytewise_simd<SimdAVX512>(
        seq, k,
        [&](
            std::size_t, auto kmers, SimdVectorValidLanes<SimdAVX512> lanes
        ) FISK_ALWAYS_INLINE_LAMBDA {
            sink.consume(kmer_value(kmers));
            do_not_optimize(lanes.bits);
        }
    );
    return sink.finalize();
}

#endif // FISK_HAS_AVX512
