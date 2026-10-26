#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_extract/packed_rolling_scalar.hpp"
#include "kmer_extract_packed_unchecked/bench.hpp"

using namespace fisk;

std::uint64_t run_var_unchecked_rolling_scalar_lsb(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_kmer_packed_unchecked_rolling_scalar(seq, k, [&](auto kmer) FISK_ALWAYS_INLINE_LAMBDA {
        sink.consume(kmer_value(kmer));
    });
    return sink.finalize();
}
