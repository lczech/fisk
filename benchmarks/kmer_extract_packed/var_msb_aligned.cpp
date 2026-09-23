#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_extract/packed.hpp"
#include "kmer_extract_packed/bench.hpp"

using namespace fisk;

std::uint64_t run_var_msb_aligned(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_kmer_packed_aligned(seq, k, [&](auto kmer) { sink.consume(kmer_value(kmer)); });
    return sink.finalize();
}
