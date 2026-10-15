#include <cstdint>
#include <string>

#include "fisk/seq_pack/seq_pack.hpp"
#include "seq_pack_validity/bench.hpp"

#if defined(FISK_HAS_NEON)

using namespace fisk;

std::uint64_t run_var_validity_acgt_lsb_direct_neon(
    std::string const& seq,
    PackedSequenceWithValidity<Encoding::kACGT, Layout::kLSB>& out
) {
    pack_sequence_with_validity(
        seq, WordEncoderDirect<Encoding::kACGT, Layout::kLSB, SimdNEON>{}, out
    );
    return 0;
}

#endif // FISK_HAS_NEON
