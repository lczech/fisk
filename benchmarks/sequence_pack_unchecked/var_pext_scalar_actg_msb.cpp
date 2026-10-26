#include <cstdint>
#include <string>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "sequence_pack_unchecked/bench.hpp"

#if defined(FISK_HAS_BMI2)

using namespace fisk;

std::uint64_t run_var_unchecked_pext_scalar_actg_msb(
    std::string const& seq,
    PackedSequence<Encoding::kACTG, Layout::kMSB>& out
) {
    pack_sequence_unchecked(seq, WordEncoderPext<Encoding::kACTG, Layout::kMSB>{}, out);
    return 0;
}

#endif // FISK_HAS_BMI2
