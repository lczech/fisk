#include <cstdint>
#include <string>

#include "fisk/seq_pack/seq_pack.hpp"
#include "seq_pack/bench.hpp"

#if defined(FISK_HAS_BMI2) && defined(FISK_HAS_AVX512)

using namespace fisk;

std::uint64_t run_var_actg_lsb_pext_avx512(
    std::string const& seq,
    PackedSequence<Encoding::kACTG, Layout::kLSB>& out
) {
    pack_sequence(seq, WordEncoderPext<Encoding::kACTG, Layout::kLSB, SimdAVX512>{}, out);
    return 0;
}

#endif // FISK_HAS_BMI2 && FISK_HAS_AVX512
