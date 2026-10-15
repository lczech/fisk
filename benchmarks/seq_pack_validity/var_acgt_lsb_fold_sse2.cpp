#include <cstdint>
#include <string>

#include "fisk/seq_pack/seq_pack.hpp"
#include "seq_pack_validity/bench.hpp"

#if defined(FISK_HAS_SSE2)

using namespace fisk;

std::uint64_t run_var_validity_acgt_lsb_fold_sse2(
    std::string const& seq,
    PackedSequenceWithValidity<Encoding::kACGT, Layout::kLSB>& out
) {
    pack_sequence_with_validity(
        seq, WordEncoderFold<Encoding::kACGT, Layout::kLSB, SimdSSE2>{}, out
    );
    return 0;
}

#endif // FISK_HAS_SSE2
