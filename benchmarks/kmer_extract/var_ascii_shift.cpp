#include <cstdint>
#include <string>

#include "fisk/core/char_encoder.hpp"
#include "fisk/kmer_extract/kmer_extract.hpp"
#include "kmer_extract/bench.hpp"

using namespace fisk;

std::uint64_t run_var_ascii_shift(std::string const& seq, std::size_t k)
{
    std::uint64_t hash = 0;
    for_each_kmer_rolling(seq, k, CharEncoderAscii<Encoding::kACGT>{}, [&](auto kmer) {
        hash += kmer_value(kmer);
    });
    return hash;
}
