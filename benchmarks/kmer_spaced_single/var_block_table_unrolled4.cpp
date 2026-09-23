#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/core/char_encoder.hpp"
#include "fisk/kmer_spaced/kmer_spaced.hpp"
#include "kmer_spaced_single/bench.hpp"

using namespace fisk;

std::uint64_t run_var_block_table_unrolled4(
    std::string const& seq,
    std::size_t k,
    BitExtractBlockTable const& mask,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_spaced_kmer(
        std::string_view(seq),
        k,
        mask,
        CharEncoderTable<Encoding::kACGT>{},
        bit_extract_block_table_unrolled<4>,
        [&](std::size_t /*mask_idx*/, std::size_t /*pos*/, std::uint64_t spaced_kmer) {
            sink.consume(spaced_kmer);
        }
    );
    return sink.finalize();
}
