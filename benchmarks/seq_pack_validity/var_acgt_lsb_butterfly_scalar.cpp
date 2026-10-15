#include <cstdint>
#include <string>

#include "fisk/seq_pack/seq_pack.hpp"
#include "seq_pack_validity/bench.hpp"

using namespace fisk;

std::uint64_t run_var_validity_acgt_lsb_butterfly_scalar(
    std::string const& seq,
    PackedSequenceWithValidity<Encoding::kACGT, Layout::kLSB>& out
) {
    pack_sequence_with_validity(
        seq, WordEncoderButterfly<Encoding::kACGT, Layout::kLSB>{}, out
    );
    return 0;
}
