#include <cstdint>
#include <string>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "sequence_pack_unchecked/bench.hpp"

using namespace fisk;

std::uint64_t run_var_unchecked_direct_scalar_acgt_lsb(
    std::string const& seq,
    PackedSequence<Encoding::kACGT, Layout::kLSB>& out
) {
    pack_sequence_unchecked(seq, WordEncoderDirect<Encoding::kACGT, Layout::kLSB>{}, out);
    return 0;
}
