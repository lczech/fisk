#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/core/char_encoder.hpp"
#include "fisk/kmer_spaced/kmer_spaced.hpp"
#include "kmer_spaced_multi/bench.hpp"

using namespace fisk;

std::uint64_t run_var_bitloop(std::string const& seq, std::size_t k, std::vector<BitExtractMask> const& masks)
{
    std::uint64_t hash = 0;
    for_each_spaced_kmer(
        std::string_view(seq), k, masks, CharEncoderTable<Encoding::kACGT>{}, bit_extract_bitloop,
        [&](std::size_t /*mask_idx*/, std::size_t /*pos*/, std::uint64_t spaced_kmer) {
            hash += spaced_kmer;
        }
    );
    return hash;
}

