#include <cstdint>
#include <string>

#include "fisk/seq_pack/seq_pack.hpp"
#include "seq_pack/bench.hpp"

using namespace fisk;

std::uint64_t run_var_acgt_lsb_butterfly(
    std::string const& seq,
    PackedSequence<Encoding::kACGT, Layout::kLSB>& out
) {
    pack_sequence(seq, WordEncoderButterfly<Encoding::kACGT, Layout::kLSB>{}, out);
    return 0;
}
