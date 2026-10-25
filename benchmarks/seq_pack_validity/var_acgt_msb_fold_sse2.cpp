#include <cstdint>
#include <string>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "seq_pack_validity/bench.hpp"

#if defined(FISK_HAS_SSE2)

using namespace fisk;

std::uint64_t run_var_validity_acgt_msb_fold_sse2(
    std::string const& seq,
    PackedSequenceWithValidity<Encoding::kACGT, Layout::kMSB>& out
) {
    pack_sequence_checked(
        seq, WordEncoderFold<Encoding::kACGT, Layout::kMSB, SimdSSE2>{}, out
    );
    return 0;
}

#endif // FISK_HAS_SSE2
