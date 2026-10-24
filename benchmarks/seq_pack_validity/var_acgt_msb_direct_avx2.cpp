#include <cstdint>
#include <string>

#include "fisk/seq_pack/seq_pack.hpp"
#include "seq_pack_validity/bench.hpp"

#if defined(FISK_HAS_AVX2)

using namespace fisk;

std::uint64_t run_var_validity_acgt_msb_direct_avx2(
    std::string const& seq,
    PackedSequenceWithValidity<Encoding::kACGT, Layout::kMSB>& out
) {
    pack_sequence_checked(
        seq, WordEncoderDirect<Encoding::kACGT, Layout::kMSB, SimdAVX2>{}, out
    );
    return 0;
}

#endif // FISK_HAS_AVX2
