#include <cstdint>
#include <string>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "seq_pack/bench.hpp"

using namespace fisk;

std::uint64_t run_var_actg_msb_butterfly_scalar(
    std::string const& seq,
    PackedSequence<Encoding::kACTG, Layout::kMSB>& out
) {
    pack_sequence_unchecked(seq, WordEncoderButterfly<Encoding::kACTG, Layout::kMSB>{}, out);
    return 0;
}
