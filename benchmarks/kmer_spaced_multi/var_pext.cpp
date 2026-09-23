#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/core/char_encoder.hpp"
#include "fisk/kmer_spaced/kmer_spaced.hpp"
#include "kmer_spaced_multi/bench.hpp"

#if defined(FISK_HAS_BMI2)

using namespace fisk;

std::uint64_t run_var_pext(
    std::string const& seq,
    std::size_t k,
    std::vector<BitExtractMask> const& masks,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_spaced_kmer(
        std::string_view(seq), k, masks, CharEncoderTable<Encoding::kACGT>{}, bit_extract_pext,
        [&](std::size_t /*mask_idx*/, std::size_t /*pos*/, std::uint64_t spaced_kmer) {
            sink.consume(spaced_kmer);
        }
    );
    return sink.finalize();
}

#endif // FISK_HAS_BMI2
