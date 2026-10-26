#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_extract/packed_bytewise_simd.hpp"
#include "kmer_extract_packed_checked/bench.hpp"

using namespace fisk;

std::uint64_t run_var_checked_bytewise_simd_narrow_width_scalar_lsb(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    auto const consume = [&](
        std::size_t, std::uint64_t v, SimdVectorValidLanes<SimdScalar> lanes
    ) FISK_ALWAYS_INLINE_LAMBDA {
        sink.consume(v);
        do_not_optimize(lanes.bits);
    };
    for_each_kmer_packed_bytewise_simd_narrow_width_<SimdScalar>(seq, k, consume);
    return sink.finalize();
}

