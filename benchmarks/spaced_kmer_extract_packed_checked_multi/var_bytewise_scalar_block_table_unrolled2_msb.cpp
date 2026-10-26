#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/spaced_kmer_extract/packed_bytewise_scalar.hpp"
#include "spaced_kmer_extract_packed_checked_multi/bench.hpp"

using namespace fisk;

std::uint64_t run_var_bytewise_scalar_block_table_unrolled2(
    CheckedMsb const& seq,
    std::vector<SpacedKmerMask<Layout::kMSB, BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_spaced_kmer_packed_checked_bytewise_scalar(
        seq,
        masks,
        bit_extract_block_table_unrolled<2>,
        [&](
            std::size_t /*pos*/, std::size_t /*mask_idx*/, auto const& spaced_kmer
        ) FISK_ALWAYS_INLINE_LAMBDA {
            sink.consume(kmer_value(spaced_kmer));
        }
    );
    return sink.finalize();
}
