#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/core/intrinsics.hpp"
#include "bit_extract_weights/bench.hpp"

using namespace fisk;

std::uint64_t run_var_byte_table(
    BitExtractWeightsBatch const& batch,
    std::size_t rounds,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for (std::size_t r = 0; r < rounds; ++r) {
        clobber_memory(batch.values.data());
        for (std::size_t i = 0; i < batch.values.size(); ++i) {
            sink.consume(bit_extract_byte_table(batch.values[i], batch.masks[i]));
        }
    }
    return sink.finalize();
}
