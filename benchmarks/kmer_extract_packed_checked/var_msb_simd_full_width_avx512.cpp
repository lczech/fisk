#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_extract/packed_simd.hpp"
#include "kmer_extract_packed_checked/bench.hpp"

#if defined(FISK_HAS_AVX512)

using namespace fisk;

std::uint64_t run_var_checked_msb_simd_full_width_avx512(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    auto const consume = [&](
        std::size_t, __m512i v, SimdVectorValidLanes<SimdAVX512> lanes
    ) FISK_ALWAYS_INLINE_LAMBDA {
        sink.consume(v);
        do_not_optimize(lanes.bits);
    };
    for_each_kmer_packed_bytewise_simd_full_width_<SimdAVX512>(seq, k, consume);
    return sink.finalize();
}

#endif // FISK_HAS_AVX512
