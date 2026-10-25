#include <cstdint>
#include <string>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "seq_pack/bench.hpp"

using namespace fisk;

std::uint64_t run_var_actg_lsb_direct_scalar(
    std::string const& seq,
    PackedSequence<Encoding::kACTG, Layout::kLSB>& out
) {
    pack_sequence_unchecked(seq, WordEncoderDirect<Encoding::kACTG, Layout::kLSB>{}, out);
    return 0;
}
