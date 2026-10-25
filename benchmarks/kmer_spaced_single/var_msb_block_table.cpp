#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/core/char_encoder.hpp"
#include "fisk/spaced_kmer_extract/ascii_rolling_scalar.hpp"
#include "kmer_spaced_single/bench.hpp"

using namespace fisk;

std::uint64_t run_var_block_table(
    std::string const& seq,
    SpacedKmerMask<Layout::kMSB, BitExtractBlockTable> const& mask,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_spaced_kmer_ascii_checked_rolling_scalar(
        std::string_view(seq),
        mask,
        CharEncoderTable<Encoding::kACGT>{},
        bit_extract_block_table,
        [&](
            std::size_t /*pos*/, std::size_t /*mask_idx*/, auto const& spaced_kmer
        ) FISK_ALWAYS_INLINE_LAMBDA {
            sink.consume(kmer_value(spaced_kmer));
        }
    );
    return sink.finalize();
}
