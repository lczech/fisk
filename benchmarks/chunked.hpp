#pragma once

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/bit_extract/simd.hpp"
#include "fisk/core/kmer_spaced.hpp"
#include "fisk/core/types.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"
#include "fisk/simd/avx2.hpp"
#include "fisk/simd/avx512.hpp"
#include "fisk/simd/neon.hpp"
#include "fisk/simd/scalar.hpp"
#include "fisk/simd/sse2.hpp"

// =================================================================================================
//     Chunked Extraction from ASCII
// =================================================================================================

/**
 * @brief The word encoder of the benchmarks of the chunked extraction from ASCII, and of the
 * packing that they are compared against, for the arch of the build.
 *
 * WordEncoderDirectDouble, the fastest word encoder in the sequence_pack suites, with AVX2 or Neon,
 * else SSE2, else WordEncoderButterfly. AVX512 is left out, as it is not measured for packing yet.
 * The encoder is independent of the arch of the k-mer loops, see kmer_extract/ascii_chunked.hpp.
 */
#if defined(FISK_HAS_AVX2)
template <fisk::Encoding E, fisk::Layout L>
using ChunkedWordEncoder = fisk::WordEncoderDirectDouble<E, L, fisk::SimdAVX2>;
#elif defined(FISK_HAS_NEON)
template <fisk::Encoding E, fisk::Layout L>
using ChunkedWordEncoder = fisk::WordEncoderDirectDouble<E, L, fisk::SimdNEON>;
#elif defined(FISK_HAS_SSE2)
template <fisk::Encoding E, fisk::Layout L>
using ChunkedWordEncoder = fisk::WordEncoderDirectDouble<E, L, fisk::SimdSSE2>;
#else
template <fisk::Encoding E, fisk::Layout L>
using ChunkedWordEncoder = fisk::WordEncoderButterfly<E, L>;
#endif

// The scratch of the chunked loops, without and with validity, and the masks and kernels of the
// spaced ones.
template <fisk::Layout L>
using ChunkedScratch = fisk::PackedSequence<fisk::Encoding::kACGT, L>;
template <fisk::Layout L>
using ChunkedCheckedScratch = fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, L>;
template <fisk::Layout L>
using PextMask = fisk::SpacedKmerMask<L, fisk::BitExtractMask>;
template <fisk::Layout L>
using ButterflyMask = fisk::SpacedKmerMask<L, fisk::BitExtractButterflyMask>;
template <fisk::Layout L, typename Simd>
using ButterflyKernel = fisk::SpacedKmerMask<L, fisk::BitExtractKernelButterfly<Simd>>;
