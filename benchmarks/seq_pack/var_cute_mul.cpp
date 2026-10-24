#include <cstdint>
#include <string>

#include "external/cute_nucleotides.hpp"
#include "seq_pack/bench.hpp"

#if defined(FISK_HAS_AVX2)

using namespace fisk;

std::uint64_t run_var_cute_mul(
    std::string const& seq,
    PackedSequence<Encoding::kACTG, Layout::kLSB>& out
) {
    cute_nucleotides::pack_sequence(seq, cute_nucleotides::encode_mul, out);
    return 0;
}

#endif // FISK_HAS_AVX2
