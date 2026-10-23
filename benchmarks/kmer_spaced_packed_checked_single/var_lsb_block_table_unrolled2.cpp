#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_spaced/packed.hpp"
#include "kmer_spaced_packed_checked_single/bench.hpp"

using namespace fisk;

std::uint64_t run_var_block_table_unrolled2(
    CheckedLsb const& seq,
    SpacedKmerMask<Layout::kLSB, BitExtractBlockTable> const& mask,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_spaced_kmer_packed_checked_aligned(
        seq,
        mask,
        bit_extract_block_table_unrolled<2>,
        [&](
            std::size_t /*pos*/, std::size_t /*mask_idx*/, auto const& spaced_kmer
        ) FISK_ALWAYS_INLINE_LAMBDA {
            sink.consume(kmer_value(spaced_kmer));
        }
    );
    return sink.finalize();
}
