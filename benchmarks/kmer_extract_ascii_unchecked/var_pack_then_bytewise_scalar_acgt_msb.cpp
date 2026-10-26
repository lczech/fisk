#include <cstdint>
#include <string>
#include <vector>

#include "fisk/kmer_extract/packed_bytewise_scalar.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "kmer_extract_ascii_unchecked/bench.hpp"

using namespace fisk;

std::uint64_t run_var_pack_then_bytewise_scalar_acgt_msb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtMsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
) {
    pack_sequence_unchecked(seq, WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>{}, scratch);
    auto sink = make_sink(sink_buffer);
    for_each_kmer_packed_unchecked_bytewise_scalar(
        scratch, k, [&](auto kmer) FISK_ALWAYS_INLINE_LAMBDA { sink.consume(kmer_value(kmer)); }
    );
    return sink.finalize();
}
