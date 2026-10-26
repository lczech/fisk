#include <cstdint>
#include <string>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "sequence_pack_unchecked/bench.hpp"

#if defined(FISK_HAS_BMI2) && defined(FISK_HAS_AVX512)

using namespace fisk;

std::uint64_t run_var_unchecked_pext_avx512_acgt_lsb(
    std::string const& seq,
    PackedSequence<Encoding::kACGT, Layout::kLSB>& out
) {
    pack_sequence_unchecked(seq, WordEncoderPext<Encoding::kACGT, Layout::kLSB, SimdAVX512>{}, out);
    return 0;
}

#endif // FISK_HAS_BMI2 && FISK_HAS_AVX512
