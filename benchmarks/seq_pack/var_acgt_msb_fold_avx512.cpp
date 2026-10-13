#include <cstdint>
#include <string>

#include "fisk/seq_pack/seq_pack.hpp"
#include "seq_pack/bench.hpp"

#if defined(FISK_HAS_AVX512)

using namespace fisk;

std::uint64_t run_var_acgt_msb_fold_avx512(
    std::string const& seq,
    PackedSequence<Encoding::kACGT, Layout::kMSB>& out
) {
    pack_sequence(seq, WordEncoderFold<Encoding::kACGT, Layout::kMSB, SimdAVX512>{}, out);
    return 0;
}

#endif // FISK_HAS_AVX512
