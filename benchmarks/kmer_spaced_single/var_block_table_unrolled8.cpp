#include <cstddef>
#include <cstdint>
#include <string>

#include "fisk/core/char_encoder.hpp"
#include "fisk/kmer_spaced/kmer_spaced.hpp"
#include "kmer_spaced_single/bench.hpp"

using namespace fisk;

std::uint64_t run_var_block_table_unrolled8(std::string const& seq, std::size_t k, BitExtractBlockTable const& mask)
{
    return compute_spaced_kmer_hash(
        seq, k, mask, CharEncoderTable<Encoding::kACGT>{}, bit_extract_block_table_unrolled<8>
    );
}
