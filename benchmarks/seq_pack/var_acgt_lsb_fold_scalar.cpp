#include <cstdint>
#include <string>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "seq_pack/bench.hpp"

using namespace fisk;

std::uint64_t run_var_acgt_lsb_fold_scalar(
    std::string const& seq,
    PackedSequence<Encoding::kACGT, Layout::kLSB>& out
) {
    pack_sequence_unchecked(seq, WordEncoderFold<Encoding::kACGT, Layout::kLSB>{}, out);
    return 0;
}
