#include <cstdint>
#include <string>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "sequence_pack_unchecked/bench.hpp"

#if defined(FISK_HAS_AVX512)

using namespace fisk;

std::uint64_t run_var_unchecked_direct_avx512_acgt_msb(
    std::string const& seq,
    PackedSequence<Encoding::kACGT, Layout::kMSB>& out
) {
    pack_sequence_unchecked(
        seq, WordEncoderDirect<Encoding::kACGT, Layout::kMSB, SimdAVX512>{}, out
    );
    return 0;
}

#endif // FISK_HAS_AVX512
