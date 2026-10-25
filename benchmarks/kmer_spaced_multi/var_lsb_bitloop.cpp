#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/core/char_encoder.hpp"
#include "fisk/spaced_kmer_extract/ascii_rolling_scalar.hpp"
#include "kmer_spaced_multi/bench.hpp"

using namespace fisk;

std::uint64_t run_var_bitloop(
    std::string const& seq,
    std::vector<SpacedKmerMask<Layout::kLSB, BitExtractMask>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_spaced_kmer_ascii_checked_rolling_scalar(
        std::string_view(seq), masks, CharEncoderTable<Encoding::kACGT>{}, bit_extract_bitloop,
        [&](
            std::size_t /*pos*/, std::size_t /*mask_idx*/, auto const& spaced_kmer
        ) FISK_ALWAYS_INLINE_LAMBDA {
            sink.consume(kmer_value(spaced_kmer));
        }
    );
    return sink.finalize();
}

