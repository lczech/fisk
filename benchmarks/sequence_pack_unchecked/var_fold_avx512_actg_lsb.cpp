#include <cstdint>
#include <string>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "sequence_pack_unchecked/bench.hpp"

#if defined(FISK_HAS_AVX512)

using namespace fisk;

std::uint64_t run_var_unchecked_fold_avx512_actg_lsb(
    std::string const& seq,
    PackedSequence<Encoding::kACTG, Layout::kLSB>& out
) {
    pack_sequence_unchecked(seq, WordEncoderFold<Encoding::kACTG, Layout::kLSB, SimdAVX512>{}, out);
    return 0;
}

#endif // FISK_HAS_AVX512
