#include <cstdint>
#include <string>

#include "fisk/seq_pack/simd.hpp"
#include "seq_pack/bench.hpp"

#if defined(FISK_HAS_AVX512)

using namespace fisk;

std::uint64_t run_var_actg_lsb_butterfly_avx512(
    std::string const& seq,
    PackedSequence<Encoding::kACTG, Layout::kLSB>& out
) {
    pack_sequence_simd(seq, WordEncoderButterflyAVX512<Encoding::kACTG, Layout::kLSB>{}, out);
    return 0;
}

#endif // FISK_HAS_AVX512
