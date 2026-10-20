#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/core/kmer_callback.hpp"
#include "fisk/seq_pack/seq_pack.hpp"
#include "fisk/core/simd_sse2.hpp"
#include "fisk/core/simd_avx2.hpp"
#include "fisk/core/simd_avx512.hpp"
#include "fisk/core/simd_neon.hpp"
#include "sink.hpp"

// Every sequence benchmarked below is packed with its validity, with an ACGT word encoder; only
// the layout varies.
using CheckedMsb = fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>;
using CheckedLsb = fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>;

// The k-mers of the valid lanes, with zero in the others: a stand-in for what a caller does with
// the valid lanes, so that they are computed.
template <typename Simd, typename Vector>
inline Vector valid_kmers_(Vector v, fisk::KmerVectorValidLanes<Simd> lanes) noexcept
{
    // Vector lane masks (all ones or zero per lane) are and-ed in; AVX512 has a mask register.
    if constexpr (requires { Simd::bitwise_and(v, lanes.value); }) {
        return Simd::bitwise_and(v, lanes.value);
    } else {
        #if defined(FISK_HAS_AVX512)
        return _mm512_maskz_mov_epi64(lanes.value, v);
        #else
        static_assert(sizeof(Simd) == 0, "Unhandled lane mask in valid_kmers_()");
        #endif
    }
}

/**
 * @brief Benchmark checked k-mer extraction from a PackedSequenceWithValidity, the same variants
 * as bench_kmer_extract_packed(), each extracting only the valid k-mers.
 *
 * Packing with validity is done once per sequence, outside the timed region. Which nucleotides are
 * invalid, and whether they come in runs, depends on the input. See the "kmer_extract_packed"
 * suite for the unchecked extraction to compare this against.
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

// Kernels compared above, each defined in its own translation unit (var_*.cpp). "aligned" and
// "rolling" cover the full k in [1, 32] range themselves; the SIMD kernels are split into
// narrow-width (k<=29) and full-width (k in [30,32]) tiers plus an undecorated dispatcher that
// picks between them at runtime, matching the library's own for_each_kmer_packed_simd()
// split.
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
std::uint64_t run_var_checked_msb_aligned(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_msb_rolling(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_lsb_aligned(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_lsb_rolling(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);

#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_checked_msb_simd_narrow_width_sse2(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_msb_simd_full_width_sse2(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_msb_simd_sse2(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_msb_simd_sse2_scalars(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_lsb_simd_narrow_width_sse2(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_lsb_simd_full_width_sse2(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_lsb_simd_sse2(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_lsb_simd_sse2_scalars(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_checked_msb_simd_narrow_width_avx2(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_msb_simd_full_width_avx2(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_msb_simd_avx2(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_msb_simd_avx2_scalars(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_lsb_simd_narrow_width_avx2(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_lsb_simd_full_width_avx2(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_lsb_simd_avx2(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_lsb_simd_avx2_scalars(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_checked_msb_simd_narrow_width_avx512(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_msb_simd_full_width_avx512(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_msb_simd_avx512(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_msb_simd_avx512_scalars(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_lsb_simd_narrow_width_avx512(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_lsb_simd_full_width_avx512(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_lsb_simd_avx512(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_lsb_simd_avx512_scalars(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

#if defined(FISK_HAS_NEON)
std::uint64_t run_var_checked_msb_simd_narrow_width_neon(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_msb_simd_full_width_neon(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_msb_simd_neon(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_msb_simd_neon_scalars(
    CheckedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_lsb_simd_narrow_width_neon(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_lsb_simd_full_width_neon(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_lsb_simd_neon(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_checked_lsb_simd_neon_scalars(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
