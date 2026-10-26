#include <cstdint>
#include <string>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "sequence_pack_unchecked/bench.hpp"

using namespace fisk;

std::uint64_t run_var_unchecked_fold_scalar_actg_msb(
    std::string const& seq,
    PackedSequence<Encoding::kACTG, Layout::kMSB>& out
) {
    pack_sequence_unchecked(seq, WordEncoderFold<Encoding::kACTG, Layout::kMSB>{}, out);
    return 0;
}
