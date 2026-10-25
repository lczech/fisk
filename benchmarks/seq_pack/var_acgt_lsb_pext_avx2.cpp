#include <cstdint>
#include <string>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "seq_pack/bench.hpp"

#if defined(FISK_HAS_BMI2) && defined(FISK_HAS_AVX2)

using namespace fisk;

std::uint64_t run_var_acgt_lsb_pext_avx2(
    std::string const& seq,
    PackedSequence<Encoding::kACGT, Layout::kLSB>& out
) {
    pack_sequence_unchecked(seq, WordEncoderPext<Encoding::kACGT, Layout::kLSB, SimdAVX2>{}, out);
    return 0;
}

#endif // FISK_HAS_BMI2 && FISK_HAS_AVX2
