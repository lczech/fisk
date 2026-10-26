#include <cstdint>
#include <string>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "sequence_pack_checked/bench.hpp"

using namespace fisk;

std::uint64_t run_var_checked_fold_scalar_acgt_msb(
    std::string const& seq,
    PackedSequenceWithValidity<Encoding::kACGT, Layout::kMSB>& out
) {
    pack_sequence_checked(
        seq, WordEncoderFold<Encoding::kACGT, Layout::kMSB>{}, out
    );
    return 0;
}
