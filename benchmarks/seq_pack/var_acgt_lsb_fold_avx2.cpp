#include <cstdint>
#include <string>

#include "fisk/seq_pack/seq_pack.hpp"
#include "seq_pack/bench.hpp"

#if defined(FISK_HAS_AVX2)

using namespace fisk;

std::uint64_t run_var_acgt_lsb_fold_avx2(
    std::string const& seq,
    PackedSequence<Encoding::kACGT, Layout::kLSB>& out
) {
    pack_sequence(seq, WordEncoderFold<Encoding::kACGT, Layout::kLSB, SimdAVX2>{}, out);
    return 0;
}

#endif // FISK_HAS_AVX2
