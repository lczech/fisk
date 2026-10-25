#include <cstdint>
#include <string>
#include <vector>

#include "fisk/kmer_extract/packed_bytewise_scalar.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "kmer_extract/bench.hpp"

using namespace fisk;

std::uint64_t run_var_acgt_lsb_pack_then_extract(
    std::string const& seq,
    std::size_t k,
    PackedAcgtLsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
) {
    pack_sequence_unchecked(seq, WordEncoderButterfly<Encoding::kACGT, Layout::kLSB>{}, scratch);
    auto sink = make_sink(sink_buffer);
    for_each_kmer_packed_unchecked_bytewise_scalar(
        scratch, k, [&](auto kmer) FISK_ALWAYS_INLINE_LAMBDA { sink.consume(kmer_value(kmer)); }
    );
    return sink.finalize();
}
