#include <cstdint>
#include <string>

#include "fisk/seq_pack/simd.hpp"
#include "seq_pack/bench.hpp"

#if defined(FISK_HAS_NEON)

using namespace fisk;

std::uint64_t run_var_acgt_lsb_butterfly_neon(std::string const& seq, PackedSequence<Encoding::kACGT, Layout::kLSB>& out)
{
    pack_sequence_simd(seq, WordEncoderButterflyNEON<Encoding::kACGT, Layout::kLSB>{}, out);
    return 0;
}

#endif // FISK_HAS_NEON
