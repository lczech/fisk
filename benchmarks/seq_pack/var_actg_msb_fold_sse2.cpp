#include <cstdint>
#include <string>

#include "fisk/seq_pack/seq_pack.hpp"
#include "seq_pack/bench.hpp"

#if defined(FISK_HAS_SSE2)

using namespace fisk;

std::uint64_t run_var_actg_msb_fold_sse2(
    std::string const& seq,
    PackedSequence<Encoding::kACTG, Layout::kMSB>& out
) {
    pack_sequence(seq, WordEncoderFold<Encoding::kACTG, Layout::kMSB, SimdSSE2>{}, out);
    return 0;
}

#endif // FISK_HAS_SSE2
