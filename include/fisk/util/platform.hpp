#pragma once

#include <bit>

namespace fisk {

// =================================================================================================
//     Host Assumptions
// =================================================================================================

// What fisk assumes about the host it is built for, asserted once here instead of in every
// header that depends on it. Included by the core headers (intrinsics.hpp, char_encoder.hpp and
// types.hpp), so that every other header picks these up transitively, whether it needs SIMD
// intrinsics or is purely scalar.

// Some of our SWAR/PEXT-based functions load several raw ASCII bytes into a single integer
// via memcpy, and rely on the first byte ending up in the least-significant position of the
// resulting value. This only holds on little-endian hosts; on a big-endian host, the same memcpy
// would silently reorder bytes within each chunk, and those would produce wrong results with no
// crash or other symptom to catch it. All of our realistic targets (x86, and ARM in its
// near-universal little-endian mode) satisfy this, so we assert it here to turn the silent failure
// into a compile error instead, should this ever be built for a genuinely big-endian target.
static_assert(
    std::endian::native == std::endian::little,
    "fisk assumes a little-endian host for its byte-to-integer packing tricks"
);

} // namespace fisk
