#include <cstdint>
#include <string>

#include "external/fastdna.hpp"
#include "sequence_pack_unchecked/bench.hpp"

using namespace fisk;

std::uint64_t run_var_fastdna_lut(
    std::string const& seq,
    PackedSequence<Encoding::kACTG, Layout::kMSB>& out
) {
    fastdna::pack_sequence_unchecked<Layout::kMSB, false, 1>(seq, out);
    return 0;
}
