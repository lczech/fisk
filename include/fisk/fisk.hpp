#pragma once

// Umbrella header that includes the entire fisk library.
// Most users will want to include only the specific headers they need instead;
// see the theme subdirectories (bit_extract/, kmer_extract/, spaced_kmer_extract/, sequence_pack/,
// core/).

// Core
#include "fisk/core/platform.hpp"
#include "fisk/core/intrinsics.hpp"
#include "fisk/core/cast.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_wide.hpp"
#include "fisk/core/kmer_spaced.hpp"
#include "fisk/core/kmer_hash.hpp"
#include "fisk/core/cpu_runtime.hpp"
#include "fisk/core/random.hpp"
#include "fisk/core/simd.hpp"
#include "fisk/core/simd_scalar.hpp"
#include "fisk/core/simd_sse2.hpp"
#include "fisk/core/simd_avx2.hpp"
#include "fisk/core/simd_avx512.hpp"
#include "fisk/core/simd_neon.hpp"
#include "fisk/core/char_encoder.hpp"
#include "fisk/core/types.hpp"

// Bit extraction
#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/bit_extract/selector.hpp"
#include "fisk/bit_extract/simd.hpp"

// Sequence packing
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"

// K-mer extraction
#include "fisk/kmer_extract/ascii_rolling_scalar.hpp"
#include "fisk/kmer_extract/ascii_rolling_avx2.hpp"
#include "fisk/kmer_extract/ascii_rolling_swar.hpp"
#include "fisk/kmer_extract/ascii_reextract_scalar.hpp"
#include "fisk/kmer_extract/ascii_chunked_scalar.hpp"
#include "fisk/kmer_extract/packed_bytewise_scalar.hpp"
#include "fisk/kmer_extract/packed_bytewise_simd.hpp"
#include "fisk/kmer_extract/packed_rolling_scalar.hpp"

// Spaced k-mer extraction
#include "fisk/spaced_kmer_extract/mask.hpp"
#include "fisk/spaced_kmer_extract/ascii_rolling_scalar.hpp"
#include "fisk/spaced_kmer_extract/ascii_rolling_simd.hpp"
#include "fisk/spaced_kmer_extract/packed_bytewise_scalar.hpp"
#include "fisk/spaced_kmer_extract/packed_bytewise_simd.hpp"
