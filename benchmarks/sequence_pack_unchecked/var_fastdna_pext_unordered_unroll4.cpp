#include <cstdint>
#include <string>

#include "external/fastdna.hpp"
#include "sequence_pack_unchecked/bench.hpp"

#if defined(FISK_HAS_BMI2)

using namespace fisk;

std::uint64_t run_var_fastdna_pext_unordered_unroll4(
    std::string const& seq,
    PackedSequence<Encoding::kACTG, Layout::kLSB>& out
) {
    fastdna::pack_sequence_unchecked<Layout::kLSB, true, 4>(seq, out);
    return 0;
}

#endif // FISK_HAS_BMI2
