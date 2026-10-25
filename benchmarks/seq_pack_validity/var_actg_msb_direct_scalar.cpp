#include <cstdint>
#include <string>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "seq_pack_validity/bench.hpp"

using namespace fisk;

std::uint64_t run_var_validity_actg_msb_direct_scalar(
    std::string const& seq,
    PackedSequenceWithValidity<Encoding::kACTG, Layout::kMSB>& out
) {
    pack_sequence_checked(
        seq, WordEncoderDirect<Encoding::kACTG, Layout::kMSB>{}, out
    );
    return 0;
}
