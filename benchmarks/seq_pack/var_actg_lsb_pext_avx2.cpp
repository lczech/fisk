#include <cstdint>
#include <string>

#include "fisk/seq_pack/seq_pack.hpp"
#include "seq_pack/bench.hpp"

#if defined(FISK_HAS_BMI2) && defined(FISK_HAS_AVX2)

using namespace fisk;

std::uint64_t run_var_actg_lsb_pext_avx2(
    std::string const& seq,
    PackedSequence<Encoding::kACTG, Layout::kLSB>& out
) {
    pack_sequence_unchecked(seq, WordEncoderPext<Encoding::kACTG, Layout::kLSB, SimdAVX2>{}, out);
    return 0;
}

#endif // FISK_HAS_BMI2 && FISK_HAS_AVX2
