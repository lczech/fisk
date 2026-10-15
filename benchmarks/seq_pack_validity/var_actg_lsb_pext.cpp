#include <cstdint>
#include <string>

#include "fisk/seq_pack/seq_pack.hpp"
#include "seq_pack_validity/bench.hpp"

#if defined(FISK_HAS_BMI2)

using namespace fisk;

std::uint64_t run_var_validity_actg_lsb_pext(
    std::string const& seq,
    PackedSequenceWithValidity<Encoding::kACTG, Layout::kLSB>& out
) {
    pack_sequence_with_validity(
        seq, WordEncoderPext<Encoding::kACTG, Layout::kLSB>{}, out
    );
    return 0;
}

#endif // FISK_HAS_BMI2
