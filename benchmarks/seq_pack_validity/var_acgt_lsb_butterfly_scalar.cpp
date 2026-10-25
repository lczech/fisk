#include <cstdint>
#include <string>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "seq_pack_validity/bench.hpp"

using namespace fisk;

std::uint64_t run_var_validity_acgt_lsb_butterfly_scalar(
    std::string const& seq,
    PackedSequenceWithValidity<Encoding::kACGT, Layout::kLSB>& out
) {
    pack_sequence_checked(
        seq, WordEncoderButterfly<Encoding::kACGT, Layout::kLSB>{}, out
    );
    return 0;
}
