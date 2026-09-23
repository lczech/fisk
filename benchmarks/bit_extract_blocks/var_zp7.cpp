#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/core/intrinsics.hpp"
#include "external/zp7.hpp"
#include "bit_extract_blocks/bench.hpp"

using namespace fisk;

std::uint64_t run_var_zp7(
    BitExtractBlocksBatch const& batch,
    std::size_t rounds,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for (std::size_t r = 0; r < rounds; ++r) {
        clobber_memory(batch.values.data());
        for (std::size_t i = 0; i < batch.values.size(); ++i) {
            sink.consume(zp7_pext_64(batch.values[i], batch.masks[i].mask));
        }
    }
    return sink.finalize();
}
