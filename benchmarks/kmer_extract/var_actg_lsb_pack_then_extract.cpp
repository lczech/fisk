#include <cstdint>
#include <string>
#include <vector>

#include "fisk/kmer_extract/packed.hpp"
#include "fisk/seq_pack/seq_pack.hpp"
#include "kmer_extract/bench.hpp"

using namespace fisk;

std::uint64_t run_var_actg_lsb_pack_then_extract(
    std::string const& seq,
    std::size_t k,
    PackedActgLsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
) {
    pack_sequence(seq, WordEncoderButterfly<Encoding::kACTG, Layout::kLSB>{}, scratch);
    auto sink = make_sink(sink_buffer);
    for_each_kmer_packed_aligned(scratch, k, [&](auto kmer) { sink.consume(kmer_value(kmer)); });
    return sink.finalize();
}
