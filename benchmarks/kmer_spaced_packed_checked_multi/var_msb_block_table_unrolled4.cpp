#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_spaced/packed.hpp"
#include "kmer_spaced_packed_checked_multi/bench.hpp"

using namespace fisk;

std::uint64_t run_var_block_table_unrolled4(
    CheckedMsb const& seq,
    std::vector<SpacedKmerMask<Layout::kMSB, BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_spaced_kmer_packed_checked_aligned(
        seq,
        masks,
        bit_extract_block_table_unrolled<4>,
        [&](
            std::size_t /*pos*/, std::size_t /*mask_idx*/, auto const& spaced_kmer
        ) FISK_ALWAYS_INLINE_LAMBDA {
            sink.consume(kmer_value(spaced_kmer));
        }
    );
    return sink.finalize();
}
