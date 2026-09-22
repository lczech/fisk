#include <cstddef>
#include <cstdint>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/core/intrinsics.hpp"
#include "external/instlatx64.hpp"
#include "bit_extract_weights/bench.hpp"

#if defined(PLATFORM_X86_64) && defined(FISK_HAS_CLMUL)

using namespace fisk;

std::uint64_t run_var_instlatx(BitExtractWeightsBatch const& batch, std::size_t rounds)
{
    std::uint64_t hash = 0;
    for (std::size_t r = 0; r < rounds; ++r) {
        clobber_memory(batch.values.data());
        for (std::size_t i = 0; i < batch.values.size(); ++i) {
            hash += pext64_emu(batch.values[i], batch.masks[i].mask);
        }
        do_not_optimize(hash);
    }
    return hash;
}

#endif // defined(PLATFORM_X86_64) && defined(FISK_HAS_CLMUL)
