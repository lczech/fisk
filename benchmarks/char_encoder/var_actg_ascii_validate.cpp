#include <cstdint>
#include <string>

#include "char_encoder/bench.hpp"
#include "fisk/core/char_encoder.hpp"

using namespace fisk;

std::uint64_t run_var_actg_ascii_validate(std::string const& seq)
{
    CharEncoderAscii<Encoding::kACTG> const encoder;
    std::uint64_t h = 0;
    for (char c : seq) {
        h += encoder(c);
    }
    return h;
}
