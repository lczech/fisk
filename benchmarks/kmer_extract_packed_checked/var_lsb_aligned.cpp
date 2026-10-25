#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_extract/packed_bytewise_scalar.hpp"
#include "kmer_extract_packed_checked/bench.hpp"

using namespace fisk;

std::uint64_t run_var_checked_lsb_aligned(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_kmer_packed_checked_bytewise_scalar(
        seq, k, [&](auto kmer) FISK_ALWAYS_INLINE_LAMBDA { sink.consume(kmer_value(kmer)); }
    );
    return sink.finalize();
}
