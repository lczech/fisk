#include <cstdint>
#include <string>

#include "fisk/seq_pack/seq_pack.hpp"
#include "seq_pack_validity/bench.hpp"

#if defined(FISK_HAS_SSSE3)

using namespace fisk;

std::uint64_t run_var_validity_actg_lsb_direct_double_sse2(
    std::string const& seq,
    PackedSequenceWithValidity<Encoding::kACTG, Layout::kLSB>& out
) {
    pack_sequence_checked(
        seq, WordEncoderDirectDouble<Encoding::kACTG, Layout::kLSB, SimdSSE2>{}, out
    );
    return 0;
}

#endif // FISK_HAS_SSSE3
