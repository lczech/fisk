#include <cstdint>
#include <string>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "seq_pack/bench.hpp"

#if defined(FISK_HAS_SSSE3)

using namespace fisk;

std::uint64_t run_var_acgt_lsb_direct_double_sse2(
    std::string const& seq,
    PackedSequence<Encoding::kACGT, Layout::kLSB>& out
) {
    pack_sequence_unchecked(
        seq, WordEncoderDirectDouble<Encoding::kACGT, Layout::kLSB, SimdSSE2>{}, out
    );
    return 0;
}

#endif // FISK_HAS_SSSE3
