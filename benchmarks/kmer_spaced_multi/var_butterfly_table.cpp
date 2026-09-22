#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/core/char_encoder.hpp"
#include "fisk/kmer_spaced/kmer_spaced.hpp"
#include "kmer_spaced_multi/bench.hpp"

using namespace fisk;

std::uint64_t run_var_butterfly_table(std::string const& seq, std::size_t k, std::vector<BitExtractButterflyTable> const& masks)
{
    return compute_spaced_kmer_hash(seq, k, masks, CharEncoderTable<Encoding::kACGT>{}, bit_extract_butterfly_table);
}
