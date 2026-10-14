#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_spaced/packed.hpp"
#include "kmer_spaced_packed_multi/bench.hpp"

using namespace fisk;

std::uint64_t run_var_block_table(
    PackedLsb const& seq,
    std::vector<SpacedKmerMask<Layout::kLSB, BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_spaced_kmer_packed(
        seq,
        masks,
        bit_extract_block_table,
        [&](std::size_t /*pos*/, std::size_t /*mask_idx*/, auto const& spaced_kmer) {
            sink.consume(kmer_value(spaced_kmer));
        }
    );
    return sink.finalize();
}
