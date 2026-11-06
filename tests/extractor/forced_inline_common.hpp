#pragma once

// The k-mer loops forced inline, as programs that use fisk get them. The test suite otherwise
// defines FISK_ALWAYS_INLINE_FOR_EACH as empty (see CMakeLists.txt), as forcing all of its many
// instantiations inline costs too much compile time. The test_forced_inline_*.cpp files undefine it
// before any fisk header, so that util/intrinsics.hpp defines it as for any other program, and
// each loop is compiled and run once with it: the algorithms of the extractors, and the loops that
// are none of them. The loops are split across these files, as each file needs a lot of memory to
// compile, in particular with sanitizers.

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "fisk/core/char_encoder.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/kmer_extract/ascii_reextract_scalar.hpp"
#include "fisk/kmer_extract/ascii_rolling_scalar.hpp"
#include "fisk/kmer_extract/ascii_rolling_simd.hpp"
#include "fisk/kmer_extract/packed_rolling_scalar.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"
#include "fisk/spaced_kmer_extract/ascii_rolling_scalar.hpp"
#include "fisk/spaced_kmer_extract/ascii_rolling_simd.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"
#include "extractor/common.hpp"
#include "extractor/kmer_common.hpp"
#include "extractor/spaced_common.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

inline constexpr auto kAcgt = Encoding::kACGT;
inline constexpr auto kMsb = Layout::kMSB;
inline constexpr auto kAscii = InputFormat::kAscii;
inline constexpr auto kPacked = InputFormat::kPacked;
inline constexpr auto kChecked = InputCheck::kChecked;
inline constexpr auto kUnchecked = InputCheck::kUnchecked;

// Calls `f` with each SIMD arch that is compiled in.
template <typename F>
inline void for_each_compiled_simd_arch(F const& f)
{
    f.template operator()<SimdScalar>();
    #if defined(FISK_HAS_SSE2)
    f.template operator()<SimdSSE2>();
    #endif
    #if defined(FISK_HAS_AVX2)
    f.template operator()<SimdAVX2>();
    #endif
    #if defined(FISK_HAS_AVX512)
    f.template operator()<SimdAVX512>();
    #endif
    #if defined(FISK_HAS_NEON)
    f.template operator()<SimdNEON>();
    #endif
}
