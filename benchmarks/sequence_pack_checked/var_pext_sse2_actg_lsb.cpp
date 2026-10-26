#include <cstdint>
#include <string>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "sequence_pack_checked/bench.hpp"

#if defined(FISK_HAS_BMI2) && defined(FISK_HAS_SSE2)

using namespace fisk;

std::uint64_t run_var_checked_pext_sse2_actg_lsb(
    std::string const& seq,
    PackedSequenceWithValidity<Encoding::kACTG, Layout::kLSB>& out
) {
    pack_sequence_checked(
        seq, WordEncoderPext<Encoding::kACTG, Layout::kLSB, SimdSSE2>{}, out
    );
    return 0;
}

#endif // FISK_HAS_BMI2 && FISK_HAS_SSE2
