#include <cstdint>
#include <string>

#include "fisk/seq_pack/seq_pack.hpp"
#include "seq_pack/bench.hpp"

#if defined(FISK_HAS_BMI2)

using namespace fisk;

std::uint64_t run_var_acgt_msb_pext(std::string const& seq, PackedSequence<Encoding::kACGT, Layout::kMSB>& out)
{
    pack_sequence(seq, WordEncoderPext<Encoding::kACGT, Layout::kMSB>{}, out);
    return 0;
}

#endif // FISK_HAS_BMI2
