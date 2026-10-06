#include <cstdint>
#include <string>
#include <vector>

#include "fisk/kmer_extract/ascii_assume_valid.hpp"
#include "kmer_extract/bench.hpp"

using namespace fisk;

std::uint64_t run_var_acgt_msb_chunked_assume_valid_scratch(
    std::string const& seq,
    std::size_t k,
    PackedAcgtMsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_kmer_ascii_assume_valid(
        seq, k, WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>{}, scratch,
        [&](auto kmer) { sink.consume(kmer_value(kmer)); }
    );
    return sink.finalize();
}
