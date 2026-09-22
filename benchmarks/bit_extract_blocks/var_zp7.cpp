#include <cstddef>
#include <cstdint>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/core/intrinsics.hpp"
#include "external/zp7.hpp"
#include "bit_extract_blocks/bench.hpp"

using namespace fisk;

std::uint64_t run_var_zp7(BitExtractBlocksBatch const& batch, std::size_t rounds)
{
    std::uint64_t hash = 0;
    for (std::size_t r = 0; r < rounds; ++r) {
        clobber_memory(batch.values.data());
        for (std::size_t i = 0; i < batch.values.size(); ++i) {
            hash += zp7_pext_64(batch.values[i], batch.masks[i].mask);
        }
        do_not_optimize(hash);
    }
    return hash;
}
