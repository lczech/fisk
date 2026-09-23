#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/seq_pack/seq_pack.hpp"
#include "sink.hpp"

// Every sequence benchmarked below is packed with an ACGT word encoder; only the layout varies.
using PackedMsb = fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>;
using PackedLsb = fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>;

/**
 * @brief Benchmark k-mer extraction directly from a PackedSequence (kmer_extract/packed.hpp),
 * across both Layout conventions and both accumulator widths (narrow: single 64-bit register,
 * k<=29; wide: 128-bit-equivalent, k<=32).
 *
 * Packing is done once per sequence, outside the timed region: real usage would already have the
 * sequence packed ahead of time, so packing cost should not count against extraction cost here.
 * See the "kmer_extract" suite for the ASCII-input baseline to compare this against.
 */
void bench_kmer_extract_packed(
    std::vector<std::string> const& sequences,
    std::size_t k_min,
    std::size_t k_max,
    std::ostream& csv_os
);

// Test all valid k-mer sizes.
void bench_kmer_extract_packed(
    std::vector<std::string> const& sequences,
    std::ostream& csv_os
);

// Kernels compared above, each defined in its own translation unit (var_*.cpp). "aligned" and
// "rolling" cover the full k in [1, 32] range themselves; the SIMD kernels are split into narrow
// (k<=29) and wide (k in [30,32]) tiers plus an undecorated dispatcher that picks between them at
// runtime, matching the library's own for_each_kmer_packed_simd_<isa>() split.
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
std::uint64_t run_var_msb_aligned(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_msb_rolling(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_lsb_aligned(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_lsb_rolling(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);

#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_msb_simd_narrow_sse2(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_msb_simd_wide_sse2(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_msb_simd_sse2(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_lsb_simd_narrow_sse2(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_lsb_simd_wide_sse2(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_lsb_simd_sse2(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_msb_simd_narrow_avx2(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_msb_simd_wide_avx2(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_msb_simd_avx2(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_lsb_simd_narrow_avx2(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_lsb_simd_wide_avx2(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_lsb_simd_avx2(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_msb_simd_narrow_avx512(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_msb_simd_wide_avx512(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_msb_simd_avx512(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_lsb_simd_narrow_avx512(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_lsb_simd_wide_avx512(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_lsb_simd_avx512(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_NEON)
std::uint64_t run_var_msb_simd_narrow_neon(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_msb_simd_wide_neon(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_msb_simd_neon(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_lsb_simd_narrow_neon(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_lsb_simd_wide_neon(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_lsb_simd_neon(
    PackedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
