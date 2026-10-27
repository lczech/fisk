#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/simd/sse2.hpp"
#include "fisk/simd/avx2.hpp"
#include "fisk/simd/avx512.hpp"
#include "fisk/simd/neon.hpp"
#include "sink.hpp"

// Every sequence benchmarked below is packed with an ACGT word encoder; only the layout varies.
using PackedMsb = fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>;
using PackedLsb = fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>;

/**
 * @brief Benchmark k-mer extraction directly from a PackedSequence (kmer_extract/packed_*.hpp),
 * across both Layout conventions and both accumulator widths (narrow-width: single 64-bit
 * register, k<=29; full-width: 128-bit-equivalent, k<=32).
 *
 * Packing is done once per sequence, outside the timed region: real usage would already have the
 * sequence packed ahead of time, so packing cost should not count against extraction cost here.
 * See the "kmer_extract_ascii_checked" and "kmer_extract_ascii_unchecked" suites for the
 * ASCII-input baseline to compare this against.
 */
void bench_kmer_extract_packed_unchecked(
    std::vector<std::string> const& sequences,
    std::size_t k_min,
    std::size_t k_max,
    std::ostream& csv_os
);

// Test all valid k-mer sizes.
void bench_kmer_extract_packed_unchecked(
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
// each one benchmarks (see fisk/util/intrinsics.hpp).
std::uint64_t run_var_unchecked_bytewise_scalar_msb(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_rolling_scalar_msb(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_scalar_lsb(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_rolling_scalar_lsb(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);

#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_unchecked_bytewise_simd_narrow_width_sse2_msb(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_full_width_sse2_msb(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_sse2_msb(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_sse2_scalars_msb(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_narrow_width_sse2_lsb(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_full_width_sse2_lsb(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_sse2_lsb(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_sse2_scalars_lsb(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_unchecked_bytewise_simd_narrow_width_avx2_msb(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_full_width_avx2_msb(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_avx2_msb(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_avx2_scalars_msb(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_narrow_width_avx2_lsb(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_full_width_avx2_lsb(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_avx2_lsb(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_avx2_scalars_lsb(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_unchecked_bytewise_simd_narrow_width_avx512_msb(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_full_width_avx512_msb(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_avx512_msb(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_avx512_scalars_msb(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_narrow_width_avx512_lsb(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_full_width_avx512_lsb(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_avx512_lsb(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_avx512_scalars_lsb(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_NEON)
std::uint64_t run_var_unchecked_bytewise_simd_narrow_width_neon_msb(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_full_width_neon_msb(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_neon_msb(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_neon_scalars_msb(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_narrow_width_neon_lsb(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_full_width_neon_lsb(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_neon_lsb(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_neon_scalars_lsb(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

std::uint64_t run_var_unchecked_bytewise_simd_narrow_width_scalar_msb(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_full_width_scalar_msb(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_scalar_msb(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_scalar_scalars_msb(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_narrow_width_scalar_lsb(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_full_width_scalar_lsb(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_scalar_lsb(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_unchecked_bytewise_simd_scalar_scalars_lsb(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
