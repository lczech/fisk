#include <cstdint>
#include <string>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "seq_pack/bench.hpp"

#if defined(FISK_HAS_SSE2)

using namespace fisk;

std::uint64_t run_var_actg_msb_butterfly_sse2(
    std::string const& seq,
    PackedSequence<Encoding::kACTG, Layout::kMSB>& out
) {
    pack_sequence_unchecked(
        seq, WordEncoderButterfly<Encoding::kACTG, Layout::kMSB, SimdSSE2>{}, out
    );
    return 0;
}

#endif // FISK_HAS_SSE2
