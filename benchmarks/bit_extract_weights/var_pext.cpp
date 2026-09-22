#include <cstddef>
#include <cstdint>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/core/intrinsics.hpp"
#include "bit_extract_weights/bench.hpp"

#if defined(FISK_HAS_BMI2)

using namespace fisk;

std::uint64_t run_var_pext(BitExtractWeightsBatch const& batch, std::size_t rounds)
{
    std::uint64_t hash = 0;
    for (std::size_t r = 0; r < rounds; ++r) {
        clobber_memory(batch.values.data());
        for (std::size_t i = 0; i < batch.values.size(); ++i) {
            hash += bit_extract_pext(batch.values[i], batch.masks[i]);
        }
        do_not_optimize(hash);
    }
    return hash;
}

#endif // FISK_HAS_BMI2
