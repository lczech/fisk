#include <cstdint>
#include <string>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "seq_pack_validity/bench.hpp"

#if defined(FISK_HAS_BMI2) && defined(FISK_HAS_SSE2)

using namespace fisk;

std::uint64_t run_var_validity_actg_msb_pext_sse2(
    std::string const& seq,
    PackedSequenceWithValidity<Encoding::kACTG, Layout::kMSB>& out
) {
    pack_sequence_checked(
        seq, WordEncoderPext<Encoding::kACTG, Layout::kMSB, SimdSSE2>{}, out
    );
    return 0;
}

#endif // FISK_HAS_BMI2 && FISK_HAS_SSE2
