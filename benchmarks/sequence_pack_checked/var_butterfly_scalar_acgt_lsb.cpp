#include <cstdint>
#include <string>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "sequence_pack_checked/bench.hpp"

using namespace fisk;

std::uint64_t run_var_checked_butterfly_scalar_acgt_lsb(
    std::string const& seq,
    PackedSequenceWithValidity<Encoding::kACGT, Layout::kLSB>& out
) {
    pack_sequence_checked(
        seq, WordEncoderButterfly<Encoding::kACGT, Layout::kLSB>{}, out
    );
    return 0;
}
