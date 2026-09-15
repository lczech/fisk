#pragma once

#include <bit>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/kmer_extract.hpp"

// Both MSB and LSB below load larger words from raw bytes via memcpy and rely on how the host
// interprets them as an integer. Both are therefore little-endian-specific.
static_assert(
    std::endian::native == std::endian::little,
    "fisk assumes a little-endian host for its byte-to-integer packing tricks"
);

// =================================================================================================
//     K-mer Extraction from a PackedSequence
// =================================================================================================

// ------------------------------------------------------------------------
//     Overview
// ------------------------------------------------------------------------

// Unlike the functions in kmer_extract.hpp, which take ASCII input and check character validity,
// the functions here read directly from a PackedSequence (seq_pack.hpp).
// They assume `seq` holds only valid, already-encoded bases, allowing for higher performance.
//
// `L` (see Layout in core/types.hpp) selects both which packed layout is read and the
// resulting k-mer's own bit convention. `kMSB` produces the left-rolling convention
// (`kmer = (kmer << 2) | code`), preserving lexicographic string order as integer order.
// `kLSB` produces the mirrored right-rolling convention (newest base in the high bits), which does
// not preserve lexicographic order but is cheaper to produce from LSB-packed data.
//
// `E` (see Encoding in core/types.hpp) is carried through unchanged: extraction only moves
// 2-bit codes around and never depends on which base a code stands for, but keeping it in the
// type prevents a sequence packed under one encoding from reaching code that expects another.
//
// Two public functions cover the full documented k in [1, 32]: for_each_kmer_packed_aligned()
// (the fast path) and for_each_kmer_packed_rolling() (a simpler reference baseline, not
// recommended for production). Both internally specialize for k <= 29 (where a single
// 64-bit accumulator suffices) vs k > 29 (needing an extra byte for spilling).

// ------------------------------------------------------------------------
//     Details
// ------------------------------------------------------------------------

// Precondition-check throws are routed through the shared throw_invalid_kmer_k_() (kmer_extract.hpp)
// rather than thrown directly, trying to keep these functions small enough for the
// compiler to treat as inlining candidates.

// do_not_optimize barriers on the k-mer emission in some of the functions below force each value
// to at least be manifested in a register, so a benchmark summing them cannot be proven reducible
// by the compiler. They emit no instructions on GCC/Clang, but their register requirements might still slightly affect regular callers.

// We experimented with several variations of the basic algorithm here, in order to find a solution
// that compiles to optimal code across most compilers and platforms. Some of the more promising
// variants are in commits
//  -f3910762f1d4a577da88de02347faf94e584a0f1
//  -313e33b54b07e87ec3a4852c65403d9ed77050b6
// but we since have cleaned up a bit, and are only keeping a final selection of variants here.

// -------------------------------------------------------------------------------------------------
//     Helpers
// -------------------------------------------------------------------------------------------------

// Shared tail for every fast-path variant below, decoding one base at a time (O(k) per k-mer)
// rather than reusing a main loop's word-width trick; only ever runs for the handful of k-mers
// after the fast path. The callback function is never passed in directly, as that would prevent
// compiler optimizations; instead, values are returned to the caller.
template <Encoding E, Layout L>
[[gnu::noinline, gnu::cold]]
std::size_t for_each_kmer_packed_tail_(
    PackedSequence<E, L> const& seq, std::size_t start_pos, std::size_t p_max,
    unsigned k32, std::array<std::uint64_t, 64>& out
) {
    auto decode_base_ = [&](std::size_t p) -> unsigned {
        std::uint8_t const byte = seq.data[p / 4];
        unsigned const in_byte = static_cast<unsigned>(p % 4);
        if constexpr (L == Layout::kMSB) {
            return (byte >> (6 - 2 * in_byte)) & 0x3u;
        } else {
            return (byte >> (2 * in_byte)) & 0x3u;
        }
    };

    std::size_t n = 0;
    for (std::size_t p = start_pos; p <= p_max; ++p) {
        std::uint64_t kmer = 0;
        for (unsigned i = 0; i < k32; ++i) {
            unsigned const code = decode_base_(p + i);
            if constexpr (L == Layout::kMSB) {
                kmer = (kmer << 2) | code;
            } else {
                kmer |= std::uint64_t{code} << (2 * i);
            }
        }
        out[n++] = kmer;
    }
    return n;
}

// =================================================================================================
//     Aligned
// =================================================================================================

// k <= 29 fits in a single 8-byte load, independently of which of the four nucleotides within
// a byte we are starting with. This "narrow" case can thus be optimized compared to the "wide"
// case below (k <= 32), which needs to read an extra byte.
//
// On some compiler/CPU combinations, calling this indirectly through for_each_kmer_packed_aligned()
// below costs a bit more than calling it directly: once that function also has to support k in
// [30, 32] by passing `func` to a separate callee, some compilers keep `func` addressable for the
// whole function rather than proving the narrow loop below never needs that, which can push its
// captured state out of a register even though this loop alone never requires it. Neither
// `[[gnu::always_inline]]` nor moving the k in [30, 32] cases into their own out-of-line function
// avoided this -- it appears to be an inherent cost of one function covering both ranges, not an
// inlining decision we can override. Measured up to ~1.5x on some platforms; not measurable on
// others. Kept as a known trade-off for for_each_kmer_packed_aligned()'s simpler single-name API.
template <Encoding E, Layout L, typename Func>
inline void for_each_kmer_packed_aligned_narrow_impl_(
    PackedSequence<E, L> const& seq, std::size_t k, Func&& func
) {
    unsigned const k32 = static_cast<unsigned>(k);
    std::uint64_t const mask = (std::uint64_t{1} << (2 * k32)) - 1u;
    std::size_t const p_max = seq.length - k;
    std::size_t const full_bytes = (p_max + 1) / 4;
    std::size_t const num_bytes = (seq.length + 3) / 4;
    std::size_t const fast_bytes = std::min(
        full_bytes, num_bytes >= 8 ? num_bytes - 8 : std::size_t{0}
    );

    for (std::size_t b = 0; b < fast_bytes; ++b) {
        std::uint64_t word;
        std::memcpy(&word, &seq.data[b], 8);
        if constexpr (L == Layout::kMSB) {
            // 58-2k is in [0,56]; adding the local shift recovers 64-2k-2*local.
            word = byte_swap_64(word) >> (58 - 2 * k32);
        }
        std::uint64_t v0, v1, v2, v3;
        if constexpr (L == Layout::kMSB) {
            v0 = (word >> 6) & mask;
            v1 = (word >> 4) & mask;
            v2 = (word >> 2) & mask;
            v3 = (word >> 0) & mask;
        } else {
            v0 = (word >> 0) & mask;
            v1 = (word >> 2) & mask;
            v2 = (word >> 4) & mask;
            v3 = (word >> 6) & mask;
        }

        do_not_optimize(v0);
        do_not_optimize(v1);
        do_not_optimize(v2);
        do_not_optimize(v3);
        func(v0);
        func(v1);
        func(v2);
        func(v3);
    }

    std::array<std::uint64_t, 64> tail_vals;
    std::size_t const tail_n = for_each_kmer_packed_tail_<E, L>(
        seq, 4 * fast_bytes, p_max, k32, tail_vals
    );
    for (std::size_t i = 0; i < tail_n; ++i) {
        func(tail_vals[i]);
    }
}

// k in [30, 32] needs a 9th margin byte beyond the main 8-byte word, which breaks the narrow
// single-shift exploit above. These three cases instead get their own specializations, where every
// shift is a compile-time constant via the K template parameter for best optimization potential.
template <Layout L, unsigned K>
inline std::uint64_t aligned_boundary_window_(
    std::uint64_t hi, std::uint64_t lo, unsigned local, std::uint64_t mask
) {
    // Shift values are compile-time constants.
    unsigned start;
    if constexpr (L == Layout::kMSB) {
        start = 128 - 2 * local - 2 * K;
    } else {
        start = 2 * local;
    }
    std::uint64_t bits;
    if (start >= 64) {
        bits = hi >> (start - 64);
    } else if (start + 2 * K <= 64) {
        bits = lo >> start;
    } else {
        bits = (lo >> start) | (hi << (64 - start));
    }
    return bits & mask;
}

template <Encoding E, Layout L, unsigned K, typename Func>
inline void for_each_kmer_packed_aligned_wide_impl_(PackedSequence<E, L> const& seq, Func&& func)
{
    static_assert(K >= 30 && K <= 32, "K must be in [30, 32]");

    std::uint64_t mask;
    if constexpr (K == 32) {
        mask = ~std::uint64_t{0};
    } else {
        mask = (std::uint64_t{1} << (2 * K)) - 1u;
    }

    std::size_t const p_max = seq.length - K;
    std::size_t const full_bytes = (p_max + 1) / 4;
    std::size_t const num_bytes  = (seq.length + 3) / 4;

    std::size_t const fast_bytes = (num_bytes >= 9) ? std::min(full_bytes, num_bytes - 9)
                                                     : std::size_t{0};

    for (std::size_t b = 0; b < fast_bytes; ++b) {
        std::uint64_t lo;
        std::memcpy(&lo, &seq.data[b], 8);
        std::uint64_t hi = static_cast<std::uint64_t>(seq.data[b + 8]);
        if constexpr (L == Layout::kMSB) {
            std::uint64_t const swapped_lo = byte_swap_64(lo);
            lo = hi << 56;
            hi = swapped_lo;
        }

        auto const v0 = aligned_boundary_window_<L, K>(hi, lo, 0, mask);
        auto const v1 = aligned_boundary_window_<L, K>(hi, lo, 1, mask);
        auto const v2 = aligned_boundary_window_<L, K>(hi, lo, 2, mask);
        auto const v3 = aligned_boundary_window_<L, K>(hi, lo, 3, mask);

        do_not_optimize(v0);
        do_not_optimize(v1);
        do_not_optimize(v2);
        do_not_optimize(v3);

        func(v0);
        func(v1);
        func(v2);
        func(v3);
    }

    std::array<std::uint64_t, 64> tail_vals;
    std::size_t const tail_n = for_each_kmer_packed_tail_<E, L>(
        seq, 4 * fast_bytes, p_max, K, tail_vals
    );
    for (std::size_t i = 0; i < tail_n; ++i) {
        func(tail_vals[i]);
    }
}

/**
 * @brief Extract all k-mers for k in [1, 32] directly from a PackedSequence, and call a
 * callback on each. The recommended, fastest variant in this file.
 *
 * For k <= 29, delegates to for_each_kmer_packed_aligned_narrow_impl_(). For k in [30, 32], where
 * an extra margin byte breaks that trick's single hoisted shift, dispatches directly (a 3-way
 * switch, not a function-pointer table) to for_each_kmer_packed_aligned_wide_impl_() instead,
 * whose shifts are all compile-time constants via its own K template parameter.
 */
template <Encoding E, Layout L, typename Func>
inline void for_each_kmer_packed_aligned(
    PackedSequence<E, L> const& seq, std::size_t k, Func&& func
) {
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.length < k) {
        return;
    }
    switch (k) {
        case 30: {
            for_each_kmer_packed_aligned_wide_impl_<E, L, 30>(seq, std::forward<Func>(func));
            break;
        }
        case 31: {
            for_each_kmer_packed_aligned_wide_impl_<E, L, 31>(seq, std::forward<Func>(func));
            break;
        }
        case 32: {
            for_each_kmer_packed_aligned_wide_impl_<E, L, 32>(seq, std::forward<Func>(func));
            break;
        }
        default: {
            for_each_kmer_packed_aligned_narrow_impl_(seq, k, std::forward<Func>(func));
            break;
        }
    }
}

// =================================================================================================
//     Rolling
// =================================================================================================

// k <= 29 fits in a single 64-bit accumulator, folded one byte at a time. Each byte's fold
// depends serially on the previous one, as opposed to the above independent loads.
// That serial dependency chain is what makes this version slower.
template <Encoding E, Layout L, typename Func>
inline void for_each_kmer_packed_rolling_narrow_impl_(
    PackedSequence<E, L> const& seq, std::size_t k, Func&& func
) {
    std::uint64_t const mask = (std::uint64_t{1} << (2 * k)) - 1u;

    std::uint64_t acc = 0;

    auto fold_byte_ = [&](std::uint8_t byte) {
        if constexpr (L == Layout::kMSB) {
            acc = (acc << 8) | byte;
        } else {
            acc = (acc >> 8) | (std::uint64_t{byte} << 56);
        }
    };

    std::size_t const num_bytes = (seq.length + 3) / 4;
    for (std::size_t b = 0; b < num_bytes; ++b) {
        fold_byte_(seq.data[b]);

        // The 4 k-mers ending within this byte, oldest (local 0) to newest (local 3).
        for (unsigned local = 0; local < 4; ++local) {
            std::size_t const e = 4 * b + local; // base end position of this k-mer
            if (e >= seq.length) {
                break;
            }
            if (e + 1 < k) {
                continue;
            }

            unsigned const s = 6 - 2 * local;
            unsigned shift;
            if constexpr (L == Layout::kMSB) {
                shift = s;
            } else {
                shift = 64 - 2 * static_cast<unsigned>(k) - s;
            }
            func((acc >> shift) & mask);
        }
    }
}

// k > 29 needs more than 64 bits of accumulator, so this variant folds into a hi:lo pair instead
// of narrow's single 64-bit `acc`; correct for any k in [1, 32], just with more bookkeeping per
// byte, which is why the narrow impl above still exists as the faster choice for k <= 29.
template <Encoding E, Layout L, typename Func>
inline void for_each_kmer_packed_rolling_wide_impl_(
    PackedSequence<E, L> const& seq, std::size_t k, Func&& func
) {
    std::uint64_t const mask = (k == 32)
        ? ~std::uint64_t{0}
        : ((std::uint64_t{1} << (2 * k)) - 1u);

    // `start` and which of window_()'s 3 cases it falls into depend only on `k` and `local`, never
    // on the byte data, so precompute them once instead of re-deriving 4 times per byte.
    struct LocalPlan
    {
        unsigned start;
        int case_id; // 0: hi only, 1: lo only, 2: spans hi and lo
    };
    std::array<LocalPlan, 4> plans;
    for (unsigned local = 0; local < 4; ++local) {
        unsigned const s = 6 - 2 * local;
        unsigned start;
        if constexpr (L == Layout::kMSB) {
            start = s;
        } else {
            start = 128 - 2 * static_cast<unsigned>(k) - s;
        }
        int case_id;
        if (start >= 64) {
            case_id = 0;
        } else if (start + 2 * k <= 64) {
            case_id = 1;
        } else {
            case_id = 2;
        }
        plans[local] = LocalPlan{start, case_id};
    }

    std::uint64_t hi = 0;
    std::uint64_t lo = 0;

    auto fold_byte_ = [&](std::uint8_t byte) {
        if constexpr (L == Layout::kMSB) {
            std::uint64_t const carry = lo >> 56;
            lo = (lo << 8) | byte;
            hi = (hi << 8) | carry;
        } else {
            std::uint64_t const carry = hi & 0xFFu;
            hi = (hi >> 8) | (std::uint64_t{byte} << 56);
            lo = (lo >> 8) | (carry << 56);
        }
    };

    auto window_ = [&](LocalPlan const& p) -> std::uint64_t {
        std::uint64_t bits;
        switch (p.case_id) {
            case 0:  bits = hi >> (p.start - 64); break;
            case 1:  bits = lo >> p.start; break;
            default: bits = (lo >> p.start) | (hi << (64 - p.start)); break;
        }
        return bits & mask;
    };

    std::size_t const num_bytes = (seq.length + 3) / 4;
    for (std::size_t b = 0; b < num_bytes; ++b) {
        fold_byte_(seq.data[b]);

        for (unsigned local = 0; local < 4; ++local) {
            std::size_t const e = 4 * b + local;
            if (e >= seq.length) {
                break;
            }
            if (e + 1 < k) {
                continue;
            }
            func(window_(plans[local]));
        }
    }
}

/**
 * @brief Extract all k-mers for k in [1, 32] from a PackedSequence via a rolling
 * accumulator, folding one byte at a time.
 *
 * Kept only as a slower algorithmic comparison baseline; prefer for_each_kmer_packed_aligned()
 * for production use.
 *
 * For k <= 29, delegates to for_each_kmer_packed_rolling_narrow_impl_() (single 64-bit
 * accumulator); for k in [30, 32], to for_each_kmer_packed_rolling_wide_impl_() (hi:lo pair)
 * instead, since a single 64-bit accumulator no longer fits.
 */
template <Encoding E, Layout L, typename Func>
inline void for_each_kmer_packed_rolling(
    PackedSequence<E, L> const& seq, std::size_t k, Func&& func
) {
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.length < k) {
        return;
    }
    if (k <= 29) {
        for_each_kmer_packed_rolling_narrow_impl_(seq, k, std::forward<Func>(func));
    } else {
        for_each_kmer_packed_rolling_wide_impl_(seq, k, std::forward<Func>(func));
    }
}
