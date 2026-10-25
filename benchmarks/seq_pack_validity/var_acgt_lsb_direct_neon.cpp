#include <cstdint>
#include <string>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "seq_pack_validity/bench.hpp"

#if defined(FISK_HAS_NEON)

using namespace fisk;

std::uint64_t run_var_validity_acgt_lsb_direct_neon(
    std::string const& seq,
    PackedSequenceWithValidity<Encoding::kACGT, Layout::kLSB>& out
) {
    pack_sequence_checked(
        seq, WordEncoderDirect<Encoding::kACGT, Layout::kLSB, SimdNEON>{}, out
    );
    return 0;
}

#endif // FISK_HAS_NEON
