#include <cstdint>
#include <string>
#include <vector>

#include "fisk/core/char_encoder.hpp"
#include "fisk/kmer_extract/kmer_extract.hpp"
#include "kmer_extract/bench.hpp"

using namespace fisk;

std::uint64_t run_var_switch_shift(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_kmer_rolling(seq, k, CharEncoderSwitch<Encoding::kACGT>{}, [&](auto kmer) {
        sink.consume(kmer_value(kmer));
    });
    return sink.finalize();
}
