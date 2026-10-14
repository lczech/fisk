#include <cstdint>
#include <string>

#include "external/fastdna.hpp"
#include "seq_pack/bench.hpp"

#if defined(FISK_HAS_BMI2)

using namespace fisk;

std::uint64_t run_var_fastdna_pext_ordered(
    std::string const& seq,
    PackedSequence<Encoding::kACTG, Layout::kMSB>& out
) {
    fastdna::pack_sequence<Layout::kMSB, true, 1>(seq, out);
    return 0;
}

#endif // FISK_HAS_BMI2
