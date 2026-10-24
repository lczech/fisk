#include <cstdint>
#include <string>

#include "fisk/seq_pack/seq_pack.hpp"
#include "seq_pack_validity/bench.hpp"

#if defined(FISK_HAS_NEON)

using namespace fisk;

std::uint64_t run_var_validity_acgt_msb_direct_double_neon(
    std::string const& seq,
    PackedSequenceWithValidity<Encoding::kACGT, Layout::kMSB>& out
) {
    pack_sequence_checked(
        seq, WordEncoderDirectDouble<Encoding::kACGT, Layout::kMSB, SimdNEON>{}, out
    );
    return 0;
}

#endif // FISK_HAS_NEON
