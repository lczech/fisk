#include <cstdint>
#include <string>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "seq_pack/bench.hpp"

#if defined(FISK_HAS_AVX512)

using namespace fisk;

std::uint64_t run_var_acgt_msb_butterfly_avx512(
    std::string const& seq,
    PackedSequence<Encoding::kACGT, Layout::kMSB>& out
) {
    pack_sequence_unchecked(
        seq, WordEncoderButterfly<Encoding::kACGT, Layout::kMSB, SimdAVX512>{}, out
    );
    return 0;
}

#endif // FISK_HAS_AVX512
