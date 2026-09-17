#pragma once

#include <algorithm>
#include <cassert>
#include <array>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <utility>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/kmer_extract.hpp"
#include "fisk/kmer_extract/packed.hpp"

namespace fisk {

// =================================================================================================
//     K-mer Extraction from a PackedSequence, SIMD
// =================================================================================================

// Per-ISA extractors for PackedSequence values. Each public dispatcher accepts k in [1, 32]
// and calls `func(vec, valid_count)` with consecutive k-mers in the ISA's native integer vector.
// Full vectors have `valid_count == lane_count`; the final vector may be zero-padded in high lanes.
// The suffixed narrow/wide functions are internal helpers used by the dispatchers.
//
// Unlike the scalar extractors in packed.hpp, these hand out raw vectors rather than Kmer values
// (core/kmer.hpp): a vector holds several k-mers at once, so the scalar wrapper does not apply to
// it, and wrapping the vector itself would buy a guarantee the ABI is not reliably asked to
// honour. These callbacks are therefore the one seam in the k-mer API where the conventions are
// carried by the PackedSequence's type rather than by the emitted value. The vector overloads
// of kmer_cast() (core/kmer.hpp) are the sanctioned way back across it.

// =================================================================================================
//     Shared tail packing
// =================================================================================================

// Packs scalar tail k-mers into native-width vectors, zero-padding the last vector.
template <std::size_t Lanes, typename LoadFn, typename Func>
inline void emit_tail_packed_(
    std::uint64_t const* vals, std::size_t n, LoadFn&& load, Func&& func
) {
    std::size_t i = 0;
    while (i < n) {
        std::size_t const chunk = std::min(Lanes, n - i);
        std::array<std::uint64_t, Lanes> buf{};
        for (std::size_t j = 0; j < chunk; ++j) {
            buf[j] = vals[i + j];
        }
        func(load(buf), chunk);
        i += chunk;
    }
}

// =================================================================================================
//     SSE2 (2 lanes)
// =================================================================================================

#if defined(FISK_HAS_SSE2)

/**
 * @brief Extracts k in [1, 29] as pairs of k-mers in __m128i registers.
 *
 * SSE2 shifts both lanes by one count, so each pair uses the same local position from adjacent
 * starting bytes.
 */
template <Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_simd_narrow_sse2_(
    PackedSequence<E, L> const& seq, std::size_t k, Func&& func
) {
    if (k == 0 || k > 29) {
        throw_invalid_kmer_k_(29);
    }
    if (seq.length < k) {
        return;
    }

    // SSE2 does not have variable shifts per lane, so instead of using the two lanes for two
    // consecutive k-mers, we use a pair of lanes for each of the four positions within two
    // consecutive bytes. The shift amount is then a per-register constant for each of the four
    // positions, and we can unpack the k-mers from the pair of bytes using these shifts.

    std::uint64_t const mask = (std::uint64_t{1} << (2 * k)) - 1u;
    unsigned const k32 = static_cast<unsigned>(k);

    std::size_t const p_max = seq.length - k;
    std::size_t const full_bytes = (p_max + 1) / 4;
    std::size_t const num_bytes  = (seq.length + 3) / 4;
    // Every fast-loop starting byte has a safe 8-byte read; paired b+1 is covered by the same bound.
    std::size_t const fast_bytes = std::min(
        full_bytes, num_bytes >= 8 ? num_bytes - 8 : std::size_t{0}
    );
    // Byte-paired main loop below consumes two starting bytes per iteration, so round the cutoff
    // down to an even count; if that excludes one trailing fully-valid byte (fast_bytes odd), it
    // is left to the scalar tail below.
    std::size_t const paired_fast_bytes = fast_bytes - (fast_bytes % 2);

    __m128i const mask_v = _mm_set1_epi64x(static_cast<std::int64_t>(mask));

    // Precomputed shift amount shared by both lanes of a byte-paired register: LSB is
    // a compile-time-shaped constant per `local`; MSB depends on runtime `k`.
    auto shift_of = [&](unsigned local) -> std::uint64_t {
        if constexpr (L == Layout::kMSB) {
            return (64u - 2u * k32) - 2u * local;
        } else {
            return 2u * local;
        }
    };
    std::uint64_t const shift0 = shift_of(0);
    std::uint64_t const shift1 = shift_of(1);
    std::uint64_t const shift2 = shift_of(2);
    std::uint64_t const shift3 = shift_of(3);

    auto extract = [&](__m128i pair, std::uint64_t shift) -> __m128i {
        __m128i const cnt = _mm_cvtsi64_si128(static_cast<std::int64_t>(shift));
        return _mm_and_si128(_mm_srl_epi64(pair, cnt), mask_v);
    };

    for (std::size_t b = 0; b < paired_fast_bytes; b += 2) {
        std::uint64_t word0, word1;
        std::memcpy(&word0, &seq.data[b],     8);
        std::memcpy(&word1, &seq.data[b + 1], 8);
        if constexpr (L == Layout::kMSB) {
            word0 = byte_swap_64(word0);
            word1 = byte_swap_64(word1);
        }
        __m128i const pair = _mm_set_epi64x(
            static_cast<std::int64_t>(word1), static_cast<std::int64_t>(word0)
        );

        __m128i const r0 = extract(pair, shift0);
        __m128i const r1 = extract(pair, shift1);
        __m128i const r2 = extract(pair, shift2);
        __m128i const r3 = extract(pair, shift3);

        // Transpose from "lane = which byte, grouped by local" back to "lane = which local,
        // grouped by byte", so k-mers are still emitted in strict sequence-position order despite
        // the byte-paired computation above. In-register shuffles only, no cross-lane traffic.
        func(_mm_unpacklo_epi64(r0, r1), std::size_t{2}); // byte b,   locals 0,1
        func(_mm_unpacklo_epi64(r2, r3), std::size_t{2}); // byte b,   locals 2,3
        func(_mm_unpackhi_epi64(r0, r1), std::size_t{2}); // byte b+1, locals 0,1
        func(_mm_unpackhi_epi64(r2, r3), std::size_t{2}); // byte b+1, locals 2,3
    }

    // Any byte excluded purely by the parity rounding above (fast_bytes odd) is still safe to
    // fast-read but has no partner to pair with. It is left to the scalar tail.
    std::array<std::uint64_t, 64> tail_vals;
    std::size_t const tail_n = for_each_kmer_packed_tail_<E, L>(
        seq, 4 * paired_fast_bytes, p_max, k32, tail_vals
    );
    emit_tail_packed_<2>(
        tail_vals.data(), tail_n,
        [](std::array<std::uint64_t, 2> const& buf) {
            return _mm_set_epi64x(
                static_cast<std::int64_t>(buf[1]), static_cast<std::int64_t>(buf[0])
            );
        },
        func
    );
}

/**
 * @brief Extracts k in [1, 32] as pairs of k-mers in __m128i registers.
 *
 * Builds boundary-spanning windows from adjacent 64-bit words before applying the SSE2 shifts.
 */
template <Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_simd_wide_sse2_(
    PackedSequence<E, L> const& seq, std::size_t k, Func&& func
) {
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.length < k) {
        return;
    }

    std::uint64_t const mask = (k == 32) ? ~std::uint64_t{0} : ((std::uint64_t{1} << (2 * k)) - 1u);
    unsigned const k32 = static_cast<unsigned>(k);

    std::size_t const p_max = seq.length - k;
    std::size_t const full_bytes = (p_max + 1) / 4;
    std::size_t const num_bytes  = (seq.length + 3) / 4;
    // The bound keeps both 8-byte loads and both extra-byte reads within the packed input.
    std::size_t const fast_bytes = (num_bytes >= 9) ? std::min(full_bytes, num_bytes - 9)
                                                     : std::size_t{0};
    // See for_each_kmer_packed_simd_narrow_sse2_()'s comment: rounds down for pairing, with any
    // resulting single trailing byte left to the scalar tail below.
    std::size_t const paired_fast_bytes = fast_bytes - (fast_bytes % 2);

    __m128i const mask_v = _mm_set1_epi64x(static_cast<std::int64_t>(mask));

    // Per-local start of the 2k-bit window in the combined 128-bit (hi:lo) value. Both lanes
    // share the same start for each byte pair.
    auto start_of = [&](unsigned local) -> std::uint64_t {
        if constexpr (L == Layout::kMSB) {
            return 128u - 2u * local - 2u * k32;
        } else {
            return 2u * local;
        }
    };
    // Per-local shift-count vectors, precomputed: cnt_lo is `start`, cnt_hi_l is `64-start`.
    // cnt_hi_r (`start-64`) is deliberately not a third precomputed invariant here: it is the
    // two's-complement negate of cnt_hi_l, cheap to derive with one vector op inside extract().
    // With 4 locals in flight at once, keeping all three as independent __m128i invariants needs
    // 12 simultaneously-live XMM registers, which exceeds SSE2's 16-register file once the mask,
    // working pair registers, and accumulator are added in, forcing the compiler to spill count
    // registers to the stack every iteration; deriving cnt_hi_r on the fly cuts that to 8
    // invariants, comfortably within budget.
    __m128i const cnt_lo0 = _mm_cvtsi64_si128(static_cast<std::int64_t>(start_of(0)));
    __m128i const cnt_lo1 = _mm_cvtsi64_si128(static_cast<std::int64_t>(start_of(1)));
    __m128i const cnt_lo2 = _mm_cvtsi64_si128(static_cast<std::int64_t>(start_of(2)));
    __m128i const cnt_lo3 = _mm_cvtsi64_si128(static_cast<std::int64_t>(start_of(3)));
    __m128i const cnt_hl0 = _mm_cvtsi64_si128(static_cast<std::int64_t>(64u - start_of(0)));
    __m128i const cnt_hl1 = _mm_cvtsi64_si128(static_cast<std::int64_t>(64u - start_of(1)));
    __m128i const cnt_hl2 = _mm_cvtsi64_si128(static_cast<std::int64_t>(64u - start_of(2)));
    __m128i const cnt_hl3 = _mm_cvtsi64_si128(static_cast<std::int64_t>(64u - start_of(3)));
    __m128i const zero_v = _mm_setzero_si128();

    // Funnel-shift emulation via three uniform-count shifts: `lo >> start` contributes the window
    // when start < 64, `hi << (64-start)` contributes it when start < 64 too (the overlapping-span
    // case), and `hi >> (start-64)` contributes it when start >= 64. `64-start` and `start-64` are
    // computed as plain std::uint64_t subtraction, which wraps on "underflow" exactly the way the
    // shift instruction out-of-range (>=64) rule zeroes the result; so whichever term does not
    // apply to a given `start` self-cancels with no branch, blend, or special-casing of
    // start==0 needed.
    auto extract = [&](__m128i lo_pair, __m128i hi_pair, __m128i cnt_lo, __m128i cnt_hi_l) -> __m128i {
        __m128i const cnt_hi_r = _mm_sub_epi64(zero_v, cnt_hi_l); // start - 64 == -(64 - start)
        __m128i const lo_contrib = _mm_srl_epi64(lo_pair, cnt_lo);
        __m128i const hi_contrib = _mm_or_si128(
            _mm_sll_epi64(hi_pair, cnt_hi_l), _mm_srl_epi64(hi_pair, cnt_hi_r)
        );
        return _mm_and_si128(_mm_or_si128(lo_contrib, hi_contrib), mask_v);
    };

    for (std::size_t b = 0; b < paired_fast_bytes; b += 2) {
        std::uint64_t lo0, lo1;
        std::memcpy(&lo0, &seq.data[b],     8);
        std::memcpy(&lo1, &seq.data[b + 1], 8);
        std::uint64_t hi0 = static_cast<std::uint64_t>(seq.data[b + 8]);
        std::uint64_t hi1 = static_cast<std::uint64_t>(seq.data[b + 9]);
        if constexpr (L == Layout::kMSB) {
            // Byte-swap each paired word before combining it with its extra byte.
            std::uint64_t const swapped_lo0 = byte_swap_64(lo0);
            std::uint64_t const swapped_lo1 = byte_swap_64(lo1);
            lo0 = hi0 << 56; hi0 = swapped_lo0;
            lo1 = hi1 << 56; hi1 = swapped_lo1;
        }
        __m128i const lo_pair = _mm_set_epi64x(
            static_cast<std::int64_t>(lo1), static_cast<std::int64_t>(lo0)
        );
        __m128i const hi_pair = _mm_set_epi64x(
            static_cast<std::int64_t>(hi1), static_cast<std::int64_t>(hi0)
        );

        __m128i const r0 = extract(lo_pair, hi_pair, cnt_lo0, cnt_hl0);
        __m128i const r1 = extract(lo_pair, hi_pair, cnt_lo1, cnt_hl1);
        __m128i const r2 = extract(lo_pair, hi_pair, cnt_lo2, cnt_hl2);
        __m128i const r3 = extract(lo_pair, hi_pair, cnt_lo3, cnt_hl3);

        func(_mm_unpacklo_epi64(r0, r1), std::size_t{2});
        func(_mm_unpacklo_epi64(r2, r3), std::size_t{2});
        func(_mm_unpackhi_epi64(r0, r1), std::size_t{2});
        func(_mm_unpackhi_epi64(r2, r3), std::size_t{2});
    }

    // Any odd trailing byte is left to the scalar tail.
    std::array<std::uint64_t, 64> tail_vals;
    std::size_t const tail_n = for_each_kmer_packed_tail_<E, L>(
        seq, 4 * paired_fast_bytes, p_max, k32, tail_vals
    );
    emit_tail_packed_<2>(
        tail_vals.data(), tail_n,
        [](std::array<std::uint64_t, 2> const& buf) {
            return _mm_set_epi64x(
                static_cast<std::int64_t>(buf[1]), static_cast<std::int64_t>(buf[0])
            );
        },
        func
    );
}

/**
 * @brief Extracts all k-mers for k in [1, 32] from a bitpacked sequence with SSE2,
 * yielding k-mers directly in SIMD vectors.
 */
template <Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_simd_sse2(
    PackedSequence<E, L> const& seq, std::size_t k, Func&& func
) {
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (k <= 29) {
        for_each_kmer_packed_simd_narrow_sse2_(seq, k, std::forward<Func>(func));
    } else {
        for_each_kmer_packed_simd_wide_sse2_(seq, k, std::forward<Func>(func));
    }
}

#endif // FISK_HAS_SSE2

// =================================================================================================
//     AVX2 (4 lanes)
// =================================================================================================

#if defined(FISK_HAS_AVX2)

/**
 * @brief Extracts k in [1, 29] as four k-mers in one __m256i register.
 *
 * AVX2's per-lane variable shift maps one byte's four local positions directly onto the lanes.
 */
template <Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_simd_narrow_avx2_(
    PackedSequence<E, L> const& seq, std::size_t k, Func&& func
) {
    if (k == 0 || k > 29) {
        throw_invalid_kmer_k_(29);
    }
    if (seq.length < k) {
        return;
    }

    std::uint64_t const mask = (std::uint64_t{1} << (2 * k)) - 1u;
    unsigned const k32 = static_cast<unsigned>(k);

    std::size_t const p_max = seq.length - k;
    std::size_t const full_bytes = (p_max + 1) / 4;
    std::size_t const num_bytes  = (seq.length + 3) / 4;
    std::size_t const fast_bytes = std::min(
        full_bytes, num_bytes >= 8 ? num_bytes - 8 : std::size_t{0}
    );

    __m256i const mask_v = _mm256_set1_epi64x(static_cast<std::int64_t>(mask));

    // Per-lane shift-count vector {shift(0), shift(1), shift(2), shift(3)}, loop-invariant.
    std::uint64_t shifts[4];
    for (unsigned local = 0; local < 4; ++local) {
        if constexpr (L == Layout::kMSB) {
            shifts[local] = (64u - 2u * k32) - 2u * local;
        } else {
            shifts[local] = 2u * local;
        }
    }
    __m256i const shift_v = _mm256_set_epi64x(
        static_cast<std::int64_t>(shifts[3]), static_cast<std::int64_t>(shifts[2]),
        static_cast<std::int64_t>(shifts[1]), static_cast<std::int64_t>(shifts[0])
    );

    for (std::size_t b = 0; b < fast_bytes; ++b) {
        std::uint64_t word;
        std::memcpy(&word, &seq.data[b], 8);
        if constexpr (L == Layout::kMSB) {
            word = byte_swap_64(word);
        }
        __m256i const bcast = _mm256_set1_epi64x(static_cast<std::int64_t>(word));
        __m256i const r = _mm256_and_si256(_mm256_srlv_epi64(bcast, shift_v), mask_v);

        func(r, std::size_t{4});
    }

    std::array<std::uint64_t, 64> tail_vals;
    std::size_t const tail_n = for_each_kmer_packed_tail_<E, L>(
        seq, 4 * fast_bytes, p_max, k32, tail_vals
    );
    emit_tail_packed_<4>(
        tail_vals.data(), tail_n,
        [](std::array<std::uint64_t, 4> const& buf) {
            return _mm256_loadu_si256(reinterpret_cast<__m256i const*>(buf.data()));
        },
        func
    );
}

/**
 * @brief Extracts k in [1, 32] as four k-mers in one __m256i register.
 *
 * Three per-lane shifts combine the two words that may contain a k-mer.
 */
template <Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_simd_wide_avx2_(
    PackedSequence<E, L> const& seq, std::size_t k, Func&& func
) {
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.length < k) {
        return;
    }

    std::uint64_t const mask = (k == 32) ? ~std::uint64_t{0} : ((std::uint64_t{1} << (2 * k)) - 1u);
    unsigned const k32 = static_cast<unsigned>(k);

    std::size_t const p_max = seq.length - k;
    std::size_t const full_bytes = (p_max + 1) / 4;
    std::size_t const num_bytes  = (seq.length + 3) / 4;
    std::size_t const fast_bytes = (num_bytes >= 9) ? std::min(full_bytes, num_bytes - 9)
                                                     : std::size_t{0};

    __m256i const mask_v = _mm256_set1_epi64x(static_cast<std::int64_t>(mask));

    std::uint64_t starts[4];
    for (unsigned local = 0; local < 4; ++local) {
        if constexpr (L == Layout::kMSB) {
            starts[local] = 128u - 2u * local - 2u * k32;
        } else {
            starts[local] = 2u * local;
        }
    }
    __m256i const lo_shift_v = _mm256_set_epi64x(
        static_cast<std::int64_t>(starts[3]), static_cast<std::int64_t>(starts[2]),
        static_cast<std::int64_t>(starts[1]), static_cast<std::int64_t>(starts[0])
    );
    __m256i const sixtyfour_v = _mm256_set1_epi64x(64);
    __m256i const hi_shift_l_v = _mm256_sub_epi64(sixtyfour_v, lo_shift_v); // 64 - start
    __m256i const hi_shift_r_v = _mm256_sub_epi64(lo_shift_v, sixtyfour_v); // start - 64

    for (std::size_t b = 0; b < fast_bytes; ++b) {
        std::uint64_t lo;
        std::memcpy(&lo, &seq.data[b], 8);
        std::uint64_t hi = static_cast<std::uint64_t>(seq.data[b + 8]);
        if constexpr (L == Layout::kMSB) {
            std::uint64_t const swapped_lo = byte_swap_64(lo);
            lo = hi << 56;
            hi = swapped_lo;
        }
        __m256i const lo_bcast = _mm256_set1_epi64x(static_cast<std::int64_t>(lo));
        __m256i const hi_bcast = _mm256_set1_epi64x(static_cast<std::int64_t>(hi));

        __m256i const lo_contrib = _mm256_srlv_epi64(lo_bcast, lo_shift_v);
        __m256i const hi_contrib = _mm256_or_si256(
            _mm256_sllv_epi64(hi_bcast, hi_shift_l_v), _mm256_srlv_epi64(hi_bcast, hi_shift_r_v)
        );
        __m256i const r = _mm256_and_si256(_mm256_or_si256(lo_contrib, hi_contrib), mask_v);

        func(r, std::size_t{4});
    }

    std::array<std::uint64_t, 64> tail_vals;
    std::size_t const tail_n = for_each_kmer_packed_tail_<E, L>(
        seq, 4 * fast_bytes, p_max, k32, tail_vals
    );
    emit_tail_packed_<4>(
        tail_vals.data(), tail_n,
        [](std::array<std::uint64_t, 4> const& buf) {
            return _mm256_loadu_si256(reinterpret_cast<__m256i const*>(buf.data()));
        },
        func
    );
}

/**
 * @brief Extracts all k-mers for k in [1, 32] from a bitpacked sequence with AVX2,
 * yielding k-mers directly in SIMD vectors.
 */
template <Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_simd_avx2(
    PackedSequence<E, L> const& seq, std::size_t k, Func&& func
) {
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (k <= 29) {
        for_each_kmer_packed_simd_narrow_avx2_(seq, k, std::forward<Func>(func));
    } else {
        for_each_kmer_packed_simd_wide_avx2_(seq, k, std::forward<Func>(func));
    }
}

#endif // FISK_HAS_AVX2

// =================================================================================================
//     AVX-512 (8 lanes)
// =================================================================================================

// AVX-512F + AVX-512BW are sufficient here; VBMI2 is optional and is not required by this path.

#if defined(FISK_HAS_AVX512)

/**
 * @brief Extracts k in [1, 29] as eight k-mers in one __m512i register.
 *
 * Two adjacent starting bytes are replicated into the register's two four-lane halves.
 */
template <Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_simd_narrow_avx512_(
    PackedSequence<E, L> const& seq, std::size_t k, Func&& func
) {
    if (k == 0 || k > 29) {
        throw_invalid_kmer_k_(29);
    }
    if (seq.length < k) {
        return;
    }

    std::uint64_t const mask = (std::uint64_t{1} << (2 * k)) - 1u;
    unsigned const k32 = static_cast<unsigned>(k);

    std::size_t const p_max = seq.length - k;
    std::size_t const full_bytes = (p_max + 1) / 4;
    std::size_t const num_bytes  = (seq.length + 3) / 4;
    // Every fast-loop starting byte has a safe 8-byte read; paired b+1 is covered by the same bound.
    std::size_t const fast_bytes = std::min(
        full_bytes, num_bytes >= 8 ? num_bytes - 8 : std::size_t{0}
    );
    // Each fast vector covers two starting bytes. Leave an odd byte for the zero-padded scalar tail
    // so every fast callback has eight valid lanes; consumers may then safely ignore valid_count
    // for associative reductions such as the benchmark hash.
    std::size_t const paired_fast_bytes = fast_bytes - (fast_bytes % 2);

    __m512i const mask_v = _mm512_set1_epi64(static_cast<std::int64_t>(mask));

    // Per-lane shift-count vector, the 4-local pattern tiled twice (same shift regardless of which
    // of the two paired bytes a lane belongs to).
    std::uint64_t shifts[4];
    for (unsigned local = 0; local < 4; ++local) {
        if constexpr (L == Layout::kMSB) {
            shifts[local] = (64u - 2u * k32) - 2u * local;
        } else {
            shifts[local] = 2u * local;
        }
    }
    __m512i const shift_v = _mm512_set_epi64(
        static_cast<std::int64_t>(shifts[3]), static_cast<std::int64_t>(shifts[2]),
        static_cast<std::int64_t>(shifts[1]), static_cast<std::int64_t>(shifts[0]),
        static_cast<std::int64_t>(shifts[3]), static_cast<std::int64_t>(shifts[2]),
        static_cast<std::int64_t>(shifts[1]), static_cast<std::int64_t>(shifts[0])
    );

    for (std::size_t b = 0; b < paired_fast_bytes; b += 2) {
        std::uint64_t word0, word1;
        std::memcpy(&word0, &seq.data[b],     8);
        std::memcpy(&word1, &seq.data[b + 1], 8);
        if constexpr (L == Layout::kMSB) {
            word0 = byte_swap_64(word0);
            word1 = byte_swap_64(word1);
        }
        __m512i const dbcast = _mm512_set_epi64(
            static_cast<std::int64_t>(word1), static_cast<std::int64_t>(word1),
            static_cast<std::int64_t>(word1), static_cast<std::int64_t>(word1),
            static_cast<std::int64_t>(word0), static_cast<std::int64_t>(word0),
            static_cast<std::int64_t>(word0), static_cast<std::int64_t>(word0)
        );
        __m512i const r = _mm512_and_si512(_mm512_srlv_epi64(dbcast, shift_v), mask_v);

        func(r, std::size_t{8});
    }

    std::array<std::uint64_t, 64> tail_vals;
    std::size_t const tail_n = for_each_kmer_packed_tail_<E, L>(
        seq, 4 * paired_fast_bytes, p_max, k32, tail_vals
    );
    emit_tail_packed_<8>(
        tail_vals.data(), tail_n,
        [](std::array<std::uint64_t, 8> const& buf) {
            return _mm512_loadu_si512(buf.data());
        },
        func
    );
}

/**
 * @brief Extracts k in [1, 32] as eight k-mers in one __m512i register.
 *
 * Two adjacent 128-bit windows are combined with three per-lane shifts.
 */
template <Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_simd_wide_avx512_(
    PackedSequence<E, L> const& seq, std::size_t k, Func&& func
) {
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.length < k) {
        return;
    }

    std::uint64_t const mask = (k == 32) ? ~std::uint64_t{0} : ((std::uint64_t{1} << (2 * k)) - 1u);
    unsigned const k32 = static_cast<unsigned>(k);

    std::size_t const p_max = seq.length - k;
    std::size_t const full_bytes = (p_max + 1) / 4;
    std::size_t const num_bytes  = (seq.length + 3) / 4;
    // The bound keeps both 8-byte loads and both extra-byte reads within the packed input.
    std::size_t const fast_bytes = (num_bytes >= 9) ? std::min(full_bytes, num_bytes - 9)
                                                     : std::size_t{0};
    // Each fast vector covers two starting bytes. Leave an odd byte for the zero-padded scalar tail
    // so every fast callback has eight valid lanes; consumers may then safely ignore valid_count
    // for associative reductions such as the benchmark hash.
    std::size_t const paired_fast_bytes = fast_bytes - (fast_bytes % 2);

    __m512i const mask_v = _mm512_set1_epi64(static_cast<std::int64_t>(mask));

    std::uint64_t starts[4];
    for (unsigned local = 0; local < 4; ++local) {
        if constexpr (L == Layout::kMSB) {
            starts[local] = 128u - 2u * local - 2u * k32;
        } else {
            starts[local] = 2u * local;
        }
    }
    __m512i const lo_shift_v = _mm512_set_epi64(
        static_cast<std::int64_t>(starts[3]), static_cast<std::int64_t>(starts[2]),
        static_cast<std::int64_t>(starts[1]), static_cast<std::int64_t>(starts[0]),
        static_cast<std::int64_t>(starts[3]), static_cast<std::int64_t>(starts[2]),
        static_cast<std::int64_t>(starts[1]), static_cast<std::int64_t>(starts[0])
    );
    __m512i const sixtyfour_v = _mm512_set1_epi64(64);
    __m512i const hi_shift_l_v = _mm512_sub_epi64(sixtyfour_v, lo_shift_v);
    __m512i const hi_shift_r_v = _mm512_sub_epi64(lo_shift_v, sixtyfour_v);

    auto window_avx512_ = [&](__m512i lo_d, __m512i hi_d) -> __m512i {
        // Counts >= 64 already zero a lane for VPSRLVQ/VPSLLVQ, including the start==0 and
        // start==64 boundaries. VPTERNLOGQ then combines the three contributions in one OR.
        __m512i const lo_contrib = _mm512_srlv_epi64(lo_d, lo_shift_v);
        __m512i const hi_l = _mm512_sllv_epi64(hi_d, hi_shift_l_v);
        __m512i const hi_r = _mm512_srlv_epi64(hi_d, hi_shift_r_v);
        __m512i const merged = _mm512_ternarylogic_epi64(lo_contrib, hi_l, hi_r, 0xFE);
        return _mm512_and_si512(merged, mask_v);
    };

    for (std::size_t b = 0; b < paired_fast_bytes; b += 2) {
        std::uint64_t lo0, lo1;
        std::memcpy(&lo0, &seq.data[b],     8);
        std::memcpy(&lo1, &seq.data[b + 1], 8);
        std::uint64_t hi0 = static_cast<std::uint64_t>(seq.data[b + 8]);
        std::uint64_t hi1 = static_cast<std::uint64_t>(seq.data[b + 9]);
        if constexpr (L == Layout::kMSB) {
            std::uint64_t const swapped_lo0 = byte_swap_64(lo0);
            std::uint64_t const swapped_lo1 = byte_swap_64(lo1);
            lo0 = hi0 << 56; hi0 = swapped_lo0;
            lo1 = hi1 << 56; hi1 = swapped_lo1;
        }
        __m512i const lo_dbcast = _mm512_set_epi64(
            static_cast<std::int64_t>(lo1), static_cast<std::int64_t>(lo1),
            static_cast<std::int64_t>(lo1), static_cast<std::int64_t>(lo1),
            static_cast<std::int64_t>(lo0), static_cast<std::int64_t>(lo0),
            static_cast<std::int64_t>(lo0), static_cast<std::int64_t>(lo0)
        );
        __m512i const hi_dbcast = _mm512_set_epi64(
            static_cast<std::int64_t>(hi1), static_cast<std::int64_t>(hi1),
            static_cast<std::int64_t>(hi1), static_cast<std::int64_t>(hi1),
            static_cast<std::int64_t>(hi0), static_cast<std::int64_t>(hi0),
            static_cast<std::int64_t>(hi0), static_cast<std::int64_t>(hi0)
        );

        __m512i const r = window_avx512_(lo_dbcast, hi_dbcast);

        func(r, std::size_t{8});
    }

    std::array<std::uint64_t, 64> tail_vals;
    std::size_t const tail_n = for_each_kmer_packed_tail_<E, L>(
        seq, 4 * paired_fast_bytes, p_max, k32, tail_vals
    );
    emit_tail_packed_<8>(
        tail_vals.data(), tail_n,
        [](std::array<std::uint64_t, 8> const& buf) {
            return _mm512_loadu_si512(buf.data());
        },
        func
    );
}

/**
 * @brief Extracts all k-mers for k in [1, 32] from a bitpacked sequence with AVX-5
 * 12, yielding k-mers directly in SIMD vectors.
 */
template <Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_simd_avx512(
    PackedSequence<E, L> const& seq, std::size_t k, Func&& func
) {
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (k <= 29) {
        for_each_kmer_packed_simd_narrow_avx512_(seq, k, std::forward<Func>(func));
    } else {
        for_each_kmer_packed_simd_wide_avx512_(seq, k, std::forward<Func>(func));
    }
}

#endif // FISK_HAS_AVX512

// =================================================================================================
//     NEON (2 lanes)
// =================================================================================================

#if defined(FISK_HAS_NEON)

/**
 * @brief Extracts k in [1, 29] as two k-mers per uint64x2_t register.
 */
template <Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_simd_narrow_neon_(
    PackedSequence<E, L> const& seq, std::size_t k, Func&& func
) {
    // Unlike SSE2, NEON's ushl (vshlq_u64) takes a genuine per-lane
    // signed shift-count vector, so one byte's 4 locals split cleanly into two registers
    // (locals 0,1 and locals 2,3), both broadcasting the same word. No byte-pairing or transpose
    // needed, results come out in strict sequence order directly.

    if (k == 0 || k > 29) {
        throw_invalid_kmer_k_(29);
    }
    if (seq.length < k) {
        return;
    }

    std::uint64_t const mask = (std::uint64_t{1} << (2 * k)) - 1u;
    unsigned const k32 = static_cast<unsigned>(k);

    std::size_t const p_max = seq.length - k;
    std::size_t const full_bytes = (p_max + 1) / 4;
    std::size_t const num_bytes  = (seq.length + 3) / 4;
    std::size_t const fast_bytes = std::min(
        full_bytes, num_bytes >= 8 ? num_bytes - 8 : std::size_t{0}
    );

    uint64x2_t const mask_v = vdupq_n_u64(mask);

    // Per-lane shift-count vectors, negated (right shift): {-shift(local0), -shift(local1)} and
    // {-shift(local2), -shift(local3)}. Loop-invariant regardless of L -- LSB's is a
    // compile-time-shaped constant, MSB's depends on runtime k -- so built once, not per byte.
    auto shift_of = [&](unsigned local) -> std::int64_t {
        std::uint64_t s;
        if constexpr (L == Layout::kMSB) {
            s = (64u - 2u * k32) - 2u * local;
        } else {
            s = 2u * local;
        }
        return -static_cast<std::int64_t>(s);
    };
    std::int64_t const neg_shift_lo[2] = { shift_of(0), shift_of(1) };
    std::int64_t const neg_shift_hi[2] = { shift_of(2), shift_of(3) };
    int64x2_t const shift_vec_lo = vld1q_s64(neg_shift_lo);
    int64x2_t const shift_vec_hi = vld1q_s64(neg_shift_hi);

    for (std::size_t b = 0; b < fast_bytes; ++b) {
        std::uint64_t word;
        std::memcpy(&word, &seq.data[b], 8);
        if constexpr (L == Layout::kMSB) {
            word = byte_swap_64(word);
        }
        uint64x2_t const bcast = vdupq_n_u64(word);

        uint64x2_t const r_lo = vandq_u64(vshlq_u64(bcast, shift_vec_lo), mask_v); // locals 0,1
        uint64x2_t const r_hi = vandq_u64(vshlq_u64(bcast, shift_vec_hi), mask_v); // locals 2,3

        func(r_lo, std::size_t{2});
        func(r_hi, std::size_t{2});
    }

    std::array<std::uint64_t, 64> tail_vals;
    std::size_t const tail_n = for_each_kmer_packed_tail_<E, L>(
        seq, 4 * fast_bytes, p_max, k32, tail_vals
    );
    emit_tail_packed_<2>(
        tail_vals.data(), tail_n,
        [](std::array<std::uint64_t, 2> const& buf) { return vld1q_u64(buf.data()); },
        func
    );
}

/**
 * @brief Extracts k in [1, 32] as two k-mers per uint64x2_t register.
 *
 */
template <Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_simd_wide_neon_(
    PackedSequence<E, L> const& seq, std::size_t k, Func&& func
) {
    // The same per-byte, no-transpose arrangement as the narrow path is used. The lo:hi
    // boundary-spanning window collapses to a single ushl each for the lo and hi contribution,
    // since ushl's signed shift count already encodes direction: shifting `lo` right by `start`
    // and `hi` "left" by `64-start` (which naturally becomes a right shift once `64-start` goes
    // negative, i.e. once start > 64) reproduces the exact same funnel-shift semantics the other
    // tiers need two differently-signed shift instructions for.

    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.length < k) {
        return;
    }

    std::uint64_t const mask = (k == 32) ? ~std::uint64_t{0} : ((std::uint64_t{1} << (2 * k)) - 1u);
    unsigned const k32 = static_cast<unsigned>(k);

    std::size_t const p_max = seq.length - k;
    std::size_t const full_bytes = (p_max + 1) / 4;
    std::size_t const num_bytes  = (seq.length + 3) / 4;
    std::size_t const fast_bytes = (num_bytes >= 9) ? std::min(full_bytes, num_bytes - 9)
                                                     : std::size_t{0};

    uint64x2_t const mask_v = vdupq_n_u64(mask);

    auto start_of = [&](unsigned local) -> std::uint64_t {
        if constexpr (L == Layout::kMSB) {
            return 128u - 2u * local - 2u * k32;
        } else {
            return 2u * local;
        }
    };
    // lo contribution: shift right by `start`, i.e. signed count -start.
    // hi contribution: shift "left" by (64-start); this is signed, so it silently becomes a right
    // shift by (start-64) once start > 64, exactly matching the hi-only regime -- see this
    // function's own docs above.
    std::int64_t const neg_start_lo[2] = {
        -static_cast<std::int64_t>(start_of(0)), -static_cast<std::int64_t>(start_of(1))
    };
    std::int64_t const neg_start_hi[2] = {
        -static_cast<std::int64_t>(start_of(2)), -static_cast<std::int64_t>(start_of(3))
    };
    std::int64_t const hi_shift_lo[2] = {
        64 - static_cast<std::int64_t>(start_of(0)), 64 - static_cast<std::int64_t>(start_of(1))
    };
    std::int64_t const hi_shift_hi[2] = {
        64 - static_cast<std::int64_t>(start_of(2)), 64 - static_cast<std::int64_t>(start_of(3))
    };
    int64x2_t const lo_cnt_lo = vld1q_s64(neg_start_lo);
    int64x2_t const lo_cnt_hi = vld1q_s64(neg_start_hi);
    int64x2_t const hi_cnt_lo = vld1q_s64(hi_shift_lo);
    int64x2_t const hi_cnt_hi = vld1q_s64(hi_shift_hi);

    for (std::size_t b = 0; b < fast_bytes; ++b) {
        std::uint64_t lo;
        std::memcpy(&lo, &seq.data[b], 8);
        std::uint64_t hi = static_cast<std::uint64_t>(seq.data[b + 8]);
        if constexpr (L == Layout::kMSB) {
            std::uint64_t const swapped_lo = byte_swap_64(lo);
            lo = hi << 56;
            hi = swapped_lo;
        }
        uint64x2_t const lo_bcast = vdupq_n_u64(lo);
        uint64x2_t const hi_bcast = vdupq_n_u64(hi);

        uint64x2_t const r_lo = vandq_u64(
            vorrq_u64(vshlq_u64(lo_bcast, lo_cnt_lo), vshlq_u64(hi_bcast, hi_cnt_lo)), mask_v
        );
        uint64x2_t const r_hi = vandq_u64(
            vorrq_u64(vshlq_u64(lo_bcast, lo_cnt_hi), vshlq_u64(hi_bcast, hi_cnt_hi)), mask_v
        );

        func(r_lo, std::size_t{2});
        func(r_hi, std::size_t{2});
    }

    std::array<std::uint64_t, 64> tail_vals;
    std::size_t const tail_n = for_each_kmer_packed_tail_<E, L>(
        seq, 4 * fast_bytes, p_max, k32, tail_vals
    );
    emit_tail_packed_<2>(
        tail_vals.data(), tail_n,
        [](std::array<std::uint64_t, 2> const& buf) { return vld1q_u64(buf.data()); },
        func
    );
}

/**
 * @brief Extracts all k-mers for k in [1, 32] from a bitpacked sequence with NEON,
 * yielding k-mers directly in SIMD vectors.
 */
template <Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_simd_neon(
    PackedSequence<E, L> const& seq, std::size_t k, Func&& func
) {
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (k <= 29) {
        for_each_kmer_packed_simd_narrow_neon_(seq, k, std::forward<Func>(func));
    } else {
        for_each_kmer_packed_simd_wide_neon_(seq, k, std::forward<Func>(func));
    }
}

#endif // FISK_HAS_NEON

} // namespace fisk
