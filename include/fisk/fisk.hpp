#pragma once

// Umbrella header that includes the entire fisk library.
// Most users will want to include only the specific headers they need instead;
// see the theme subdirectories (bit_extract/, kmer_extract/, kmer_spaced/, core/).

// Core
#include "fisk/core/intrinsics.hpp"
#include "fisk/core/cpu_runtime.hpp"
#include "fisk/core/random.hpp"
#include "fisk/core/seq_enc.hpp"

// Bit extraction
#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/bit_extract/adaptive.hpp"
#include "fisk/bit_extract/instlatx64.hpp"
#include "fisk/bit_extract/selector.hpp"
#include "fisk/bit_extract/simd.hpp"
#include "fisk/bit_extract/zp7.hpp"

// K-mer extraction
#include "fisk/kmer_extract/kmer_extract.hpp"
#include "fisk/kmer_extract/simd.hpp"

// Spaced k-mers
#include "fisk/kmer_spaced/kmer_spaced.hpp"
#include "fisk/kmer_spaced/simd.hpp"
#include "fisk/kmer_spaced/selector.hpp"
#include "fisk/kmer_spaced/fused_avx2.hpp"
