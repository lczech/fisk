#include <cstdint>
#include <string>
#include <vector>

#include "char_encoder/bench.hpp"
#include "fisk/core/char_encoder.hpp"

using namespace fisk;

std::uint64_t run_var_acgt_ascii_assume_valid(
    std::string const& seq,
    std::vector<std::uint64_t>& sink_buffer
) {
    CharEncoderAscii<Encoding::kACGT, InputValidity::kAssumeValid> const encoder;
    auto sink = make_sink(sink_buffer);
    for (char c : seq) {
        sink.consume(encoder(c));
    }
    return sink.finalize();
}
