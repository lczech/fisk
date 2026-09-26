#include <cstdint>
#include <string>
#include <vector>

#include "fisk/kmer_extract/ascii_assume_valid.hpp"
#include "kmer_extract/bench.hpp"

using namespace fisk;

std::uint64_t run_var_acgt_msb_chunked_assume_valid(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_kmer_ascii_assume_valid(
        seq, k, WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>{},
        [&](auto kmer) { sink.consume(kmer_value(kmer)); }
    );
    return sink.finalize();
}
