#include <cstdint>
#include <string>

#include "char_encoder/bench.hpp"
#include "fisk/core/char_encoder.hpp"

using namespace fisk;

std::uint64_t run_var_acgt_ascii_assume_valid(std::string const& seq)
{
    CharEncoderAscii<Encoding::kACGT, InputValidity::kAssumeValid> const encoder;
    std::uint64_t h = 0;
    for (char c : seq) {
        h += encoder(c);
    }
    return h;
}
