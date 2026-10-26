#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/core/kmer_callback.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/core/simd_sse2.hpp"
#include "fisk/core/simd_avx2.hpp"
#include "fisk/core/simd_avx512.hpp"
#include "fisk/core/simd_neon.hpp"
#include "sink.hpp"

// Every sequence benchmarked below is packed with its validity, with an ACGT word encoder; only
// the layout varies.
using CheckedMsb = fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>;
using CheckedLsb = fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>;

// The variants that hand out vectors sum them whole, valid lanes or not, as a caller takes the
// vectors with their valid lanes as they are, and keep the valid lanes with a do_not_optimize()
// barrier, so that they are computed. Selecting the valid lanes for the sum, for example with
// SimdVectorValidLanes::lane_mask(), would add work that callers do not need. The sums of the
// vector variants hence only match within an arch, see bench_kmer_extract_packed_checked().

/**
 * @brief Benchmark checked k-mer extraction from a PackedSequenceWithValidity, the same variants
 * as bench_kmer_extract_packed_unchecked(), each extracting only the valid k-mers.
 *
 * Packing with validity is done once per sequence, outside the timed region. Which nucleotides are
 * invalid, and whether they come in runs, depends on the input. See the
 * "kmer_extract_packed_unchecked" suite for the unchecked extraction to compare this against.
 */
void bench_kmer_extract_packed_checked(
    std::vector<std::string> const& sequences,
    std::size_t k_min,
    std::size_t k_max,
    std::ostream& csv_os
);

// Test all valid k-mer sizes.
void bench_kmer_extract_packed_checked(
    std::vector<std::string> const& sequences,
    std::ostream& csv_os
);

// Kernels compared above, each defined in its own translation unit (var_*.cpp).
// "bytewise_scalar" and "rolling_scalar" cover the full k in [1, 32] range themselves; the SIMD
// kernels are split into narrow-width (k<=29) and full-width (k in [30,32]) tiers plus an
// undecorated dispatcher that picks between them at runtime, matching the library's own
// for_each_kmer_packed_unchecked_bytewise_simd() split.
//
// Each one constructs its own local Sink (see sink.hpp) via make_sink(sink_buffer), so that Sum's
// accumulation can still auto-vectorize the way the original manual `acc = _mm256_add_epi64(...)`
// accumulation did before -- see the matching comment in bit_extract_weights/bench.hpp for why a
// Sink built elsewhere and passed in by reference would defeat that. The SIMD kernels hand their
// callback a whole register per k-mer batch rather than a single value -- that register is what
// they exist to produce for a caller to use downstream, so Sink::consume() is overloaded per
// register width (see sink.hpp) rather than extracting lanes here, which would benchmark work
// these extractors were never meant to do. Guarded to match the ISA availability of the extractor
// each one benchmarks (see fisk/core/intrinsics.hpp).
std::uint64_t run_var_checked_bytewise_scalar_msb(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_rolling_scalar_msb(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_scalar_lsb(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_rolling_scalar_lsb(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);

#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_checked_bytewise_simd_narrow_width_sse2_msb(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_full_width_sse2_msb(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_sse2_msb(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_sse2_scalars_msb(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_narrow_width_sse2_lsb(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_full_width_sse2_lsb(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_sse2_lsb(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_sse2_scalars_lsb(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_checked_bytewise_simd_narrow_width_avx2_msb(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_full_width_avx2_msb(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_avx2_msb(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_avx2_scalars_msb(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_narrow_width_avx2_lsb(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_full_width_avx2_lsb(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_avx2_lsb(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_avx2_scalars_lsb(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_checked_bytewise_simd_narrow_width_avx512_msb(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_full_width_avx512_msb(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_avx512_msb(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_avx512_scalars_msb(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_narrow_width_avx512_lsb(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_full_width_avx512_lsb(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_avx512_lsb(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_avx512_scalars_lsb(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_NEON)
std::uint64_t run_var_checked_bytewise_simd_narrow_width_neon_msb(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_full_width_neon_msb(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_neon_msb(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_neon_scalars_msb(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_narrow_width_neon_lsb(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_full_width_neon_lsb(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_neon_lsb(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_neon_scalars_lsb(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

std::uint64_t run_var_checked_bytewise_simd_narrow_width_scalar_msb(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_full_width_scalar_msb(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_scalar_msb(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_scalar_scalars_msb(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_narrow_width_scalar_lsb(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_full_width_scalar_lsb(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_scalar_lsb(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_bytewise_simd_scalar_scalars_lsb(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
