#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_extract/packed_bytewise_simd.hpp"
#include "kmer_extract_packed_unchecked/bench.hpp"

using namespace fisk;

std::uint64_t run_var_unchecked_bytewise_simd_narrow_width_scalar_lsb(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_kmer_packed_bytewise_simd_narrow_width_<SimdScalar>(
        seq, k,
        [&](std::size_t, std::uint64_t v, std::size_t) FISK_ALWAYS_INLINE_LAMBDA {
            sink.consume(v);
        }
    );
    return sink.finalize();
}

