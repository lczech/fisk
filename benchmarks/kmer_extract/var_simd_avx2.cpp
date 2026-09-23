#include <cstdint>
#include <string>
#include <vector>

#include "fisk/kmer_extract/simd.hpp"
#include "kmer_extract/bench.hpp"

using namespace fisk;

std::uint64_t run_var_simd_avx2(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_kmer_simd(seq, k, [&](auto kmer) { sink.consume(kmer_value(kmer)); });
    return sink.finalize();
}
