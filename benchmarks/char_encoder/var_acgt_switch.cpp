#include <cstdint>
#include <string>

#include "char_encoder/bench.hpp"
#include "fisk/core/char_encoder.hpp"

using namespace fisk;

std::uint64_t run_var_acgt_switch(std::string const& seq)
{
    CharEncoderSwitch<Encoding::kACGT> const encoder;
    std::uint64_t h = 0;
    for (char c : seq) {
        h += encoder(c);
    }
    return h;
}
