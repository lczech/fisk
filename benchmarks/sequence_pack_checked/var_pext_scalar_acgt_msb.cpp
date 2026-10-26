#include <cstdint>
#include <string>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "sequence_pack_checked/bench.hpp"

#if defined(FISK_HAS_BMI2)

using namespace fisk;

std::uint64_t run_var_checked_pext_scalar_acgt_msb(
    std::string const& seq,
    PackedSequenceWithValidity<Encoding::kACGT, Layout::kMSB>& out
) {
    pack_sequence_checked(
        seq, WordEncoderPext<Encoding::kACGT, Layout::kMSB>{}, out
    );
    return 0;
}

#endif // FISK_HAS_BMI2
