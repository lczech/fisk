#include <cstdint>
#include <string>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "sequence_pack_unchecked/bench.hpp"

using namespace fisk;

std::uint64_t run_var_unchecked_butterfly_scalar_actg_lsb(
    std::string const& seq,
    PackedSequence<Encoding::kACTG, Layout::kLSB>& out
) {
    pack_sequence_unchecked(seq, WordEncoderButterfly<Encoding::kACTG, Layout::kLSB>{}, out);
    return 0;
}
