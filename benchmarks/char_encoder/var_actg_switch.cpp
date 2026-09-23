#include <cstdint>
#include <string>
#include <vector>

#include "char_encoder/bench.hpp"
#include "fisk/core/char_encoder.hpp"

using namespace fisk;

std::uint64_t run_var_actg_switch(std::string const& seq, std::vector<std::uint64_t>& sink_buffer)
{
    CharEncoderSwitch<Encoding::kACTG> const encoder;
    auto sink = make_sink(sink_buffer);
    for (char c : seq) {
        sink.consume(encoder(c));
    }
    return sink.finalize();
}
