#include <cstddef>
#include <cstdint>
#include <string>

#include "fisk/core/char_encoder.hpp"
#include "fisk/kmer_spaced/kmer_spaced.hpp"
#include "kmer_spaced_single/bench.hpp"

using namespace fisk;

std::uint64_t run_var_bitloop(std::string const& seq, std::size_t k, BitExtractMask const& mask)
{
    return compute_spaced_kmer_hash(seq, k, mask, CharEncoderTable<Encoding::kACGT>{}, bit_extract_bitloop);
}

