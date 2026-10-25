#include <cstdint>
#include <string>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "seq_pack/bench.hpp"

#if defined(FISK_HAS_SSE2)

using namespace fisk;

std::uint64_t run_var_actg_lsb_direct_sse2(
    std::string const& seq,
    PackedSequence<Encoding::kACTG, Layout::kLSB>& out
) {
    pack_sequence_unchecked(seq, WordEncoderDirect<Encoding::kACTG, Layout::kLSB, SimdSSE2>{}, out);
    return 0;
}

#endif // FISK_HAS_SSE2
