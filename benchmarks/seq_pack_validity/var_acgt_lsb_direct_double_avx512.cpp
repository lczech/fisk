#include <cstdint>
#include <string>

#include "fisk/seq_pack/seq_pack.hpp"
#include "seq_pack_validity/bench.hpp"

#if defined(FISK_HAS_AVX512)

using namespace fisk;

std::uint64_t run_var_validity_acgt_lsb_direct_double_avx512(
    std::string const& seq,
    PackedSequenceWithValidity<Encoding::kACGT, Layout::kLSB>& out
) {
    pack_sequence_checked(
        seq, WordEncoderDirectDouble<Encoding::kACGT, Layout::kLSB, SimdAVX512>{}, out
    );
    return 0;
}

#endif // FISK_HAS_AVX512
