#include <cstdint>
#include <string>

#include "external/cute_nucleotides.hpp"
#include "sequence_pack_unchecked/bench.hpp"

#if defined(FISK_HAS_AVX2)

using namespace fisk;

std::uint64_t run_var_cute_movemask(
    std::string const& seq,
    PackedSequence<Encoding::kACTG, Layout::kLSB>& out
) {
    cute_nucleotides::pack_sequence_unchecked(seq, cute_nucleotides::encode_movemask, out);
    return 0;
}

#endif // FISK_HAS_AVX2
