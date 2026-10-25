#include <cstdint>
#include <string>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "seq_pack/bench.hpp"

#if defined(FISK_HAS_NEON)

using namespace fisk;

std::uint64_t run_var_actg_msb_fold_neon(
    std::string const& seq,
    PackedSequence<Encoding::kACTG, Layout::kMSB>& out
) {
    pack_sequence_unchecked(seq, WordEncoderFold<Encoding::kACTG, Layout::kMSB, SimdNEON>{}, out);
    return 0;
}

#endif // FISK_HAS_NEON
