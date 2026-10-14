#include <cstdint>
#include <string>

#include "external/fastdna.hpp"
#include "seq_pack/bench.hpp"

#if defined(FISK_HAS_BMI2)

using namespace fisk;

std::uint64_t run_var_fastdna_pext_unordered_unroll2(
    std::string const& seq,
    PackedSequence<Encoding::kACTG, Layout::kLSB>& out
) {
    fastdna::pack_sequence<Layout::kLSB, true, 2>(seq, out);
    return 0;
}

#endif // FISK_HAS_BMI2
