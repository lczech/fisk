#include <cstdint>
#include <string>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "sequence_pack_checked/bench.hpp"

#if defined(FISK_HAS_BMI2) && defined(FISK_HAS_AVX2)

using namespace fisk;

std::uint64_t run_var_checked_pext_avx2_actg_msb(
    std::string const& seq,
    PackedSequenceWithValidity<Encoding::kACTG, Layout::kMSB>& out
) {
    pack_sequence_checked(
        seq, WordEncoderPext<Encoding::kACTG, Layout::kMSB, SimdAVX2>{}, out
    );
    return 0;
}

#endif // FISK_HAS_BMI2 && FISK_HAS_AVX2
