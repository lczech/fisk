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
#include "fisk/core/seq_enc.hpp"
#include "fisk/kmer_extract/kmer_extract.hpp"

// Both Msb and Lsb below load larger words from raw bytes via memcpy and rely on how the host
// interprets them as an integer. Both are therefore little-endian-specific.
static_assert(
    std::endian::native == std::endian::little,
    "fisk assumes a little-endian host for its byte-to-integer packing tricks"
);

// =================================================================================================
//     K-mer Extraction from a Packed TwoBitSequence
// =================================================================================================

// ------------------------------------------------------------------------
//     Overview
// ------------------------------------------------------------------------

// Unlike the functions in kmer_extract.hpp, which take ASCII input and check character validity,
// the functions here read directly from an already-packed TwoBitSequence (seq_pack.hpp).
// They assume `seq` holds only valid, already-encoded bases, allowing for higher performance.
//
// `Order` (see BitOrder in core/seq_enc.hpp) selects both which packed layout is read and the
// resulting k-mer's own bit convention. `Msb` produces the left-rolling convention
// (`kmer = (kmer << 2) | code`), preserving lexicographic string order as integer order.
// `Lsb` produces the mirrored right-rolling convention (newest base in the high bits), which does
// not preserve lexicographic order but is cheaper to produce from Lsb-packed data.
//
// Variants: *_narrow covers k in [1, 29]; *_wide covers the full documented k in [1, 32].
// This is because the narrow variants can use a single 64-bit accumulator for the k-mer,
// while the wide variants must read an additional margin byte of the input data to fully extract
// the 64 bits of a k-mer. The narrow variants therefore tend to be faster and smaller,
// and should be preferred when k <= 29.

// ------------------------------------------------------------------------
//     Details
// ------------------------------------------------------------------------

// Precondition-check throws are routed through the shared throw_invalid_kmer_k_() (kmer_extract.hpp)
// rather than thrown directly, keeping these otherwise branch-free functions small enough for the
// compiler to treat as inlining candidates.

// do_not_optimize barriers on the k-mer emission in some of the functions below force each value
// to at least be manifested in a register, so a benchmark summing them cannot be proven reducible
// by the compiler. They emit no instructions on GCC/Clang, but their register requirements can
// affect optimization for real callers too. Keep all four before the callbacks for comparison.

// We experimented with several variations of the basic algorithm here, in order to find a solution
// that compiles to optimal code across all compilers and platforms. Some of the more promosing
// variants are in commit f3910762f1d4a577da88de02347faf94e584a0f1, but we since have cleaned up
// a bit, and are only keeping a final selection of reasoanable variants here.

// =================================================================================================
//     Narrow (1 <= k <= 29) K-mer Extraction
// =================================================================================================

// -------------------------------------------------------------------------------------------------
//     Helpers
// -------------------------------------------------------------------------------------------------

// The tail past the main loop fast_bytes cutoff (below) is handled by this out-of-line helper
// rather than inline, so that functions stay small enough for the compiler to inline them.
// Crucially, the callback function is never passed to this helper, as that would prevent the
// compiler from proving that values can stay in a register, forcing a memory round trip on
// every iteration. The helper instead collects its (at most 32, see below) values into `out`.
template <BitOrder Order>
[[gnu::noinline, gnu::cold]]
std::size_t for_each_kmer_packed_narrow_tail_(
    TwoBitSequence<Order> const& seq, std::size_t fast_bytes, std::size_t num_bytes,
    std::size_t p_max, unsigned k32, std::uint64_t mask, std::array<std::uint64_t, 32>& out
) {
    auto extract_ = [&](std::uint64_t word, unsigned local) -> std::uint64_t {
        if constexpr (Order == BitOrder::Msb) {
            return (word >> (64 - 2 * local - 2 * k32)) & mask;
        } else {
            return (word >> (2 * local)) & mask;
        }
    };

    // Every byte from fast_bytes to the end of real content -- read only as many bytes as
    // actually remain (never past `seq.data`), and bounds-check each of the 4 start positions
    // individually. At most 8 bytes (7 fully-valid-but-excluded, see fast_bytes's own docs below,
    // plus 1 partial) ever reach here, hence `out`'s fixed size of 8*4=32.
    std::size_t n = 0;
    for (std::size_t b = fast_bytes; b < num_bytes; ++b) {
        std::uint64_t word = 0;
        std::memcpy(&word, &seq.data[b], std::min<std::size_t>(8, num_bytes - b));
        if constexpr (Order == BitOrder::Msb) {
            word = byte_swap_64(word);
        }
        for (unsigned local = 0; local < 4; ++local) {
            if (4 * b + local > p_max) {
                break;
            }
            out[n++] = extract_(word, local);
        }
    }
    return n;
}

// -------------------------------------------------------------------------------------------------
//     Blockwise
// -------------------------------------------------------------------------------------------------

/**
 * @brief Extract all k-mers for k in [1, 29] directly from a packed TwoBitSequence, and call a
 * callback on each, using blockwise processing of the sequence.
 */
template <BitOrder Order, typename Func>
inline void for_each_kmer_packed_narrow_blockwise(
    TwoBitSequence<Order> const& seq, std::size_t k, Func&& func
) {
    if (k == 0 || k > 29) {
        throw_invalid_kmer_k_(29);
    }
    if (seq.length < k) {
        return;
    }

    std::uint64_t const mask = (std::uint64_t{1} << (2 * k)) - 1u;
    unsigned const k32 = static_cast<unsigned>(k);

    // Last valid start position for a k-mer of length k.
    std::size_t const p_max = seq.length - k;

    // Number of leading bytes whose all 4 start positions (4b .. 4b+3) satisfy p<=p_max. At most
    // one byte (the one containing p_max) has some but not all 4 positions valid; every byte
    // before it is fully valid, every byte after it is fully invalid.
    std::size_t const full_bytes = (p_max + 1) / 4;
    std::size_t const num_bytes  = (seq.length + 3) / 4;

    // TwoBitSequence carries no trailing padding (core/seq_enc.hpp), so an unconditional 8-byte
    // read is only safe while it stays within real content, i.e. while b+8<=num_bytes. fast_bytes
    // is the largest byte count that is both fully valid and safe to read at that width; anything
    // from there to num_bytes falls through to the boundary tail loop instead.
    std::size_t const fast_bytes = std::min(
        full_bytes, num_bytes >= 8 ? num_bytes - 8 : std::size_t{0}
    );

    // Loads a fresh 8-byte word per byte, to emit the 4 k-mers starting within each byte:
    // the k-mer starting at relative base `local` (0..3) only needs bases forward of it,
    // all already present in the just-loaded word for any k<=29.
    auto extract_ = [&](std::uint64_t word, unsigned local) -> std::uint64_t {
        if constexpr (Order == BitOrder::Msb) {
            return (word >> (64 - 2 * local - 2 * k32)) & mask;
        } else {
            return (word >> (2 * local)) & mask;
        }
    };

    // Main loop: every byte here is fully in range and far enough from the end of `seq.data` for
    // an 8-byte read to stay inside real content, so no bounds check and no early return needed.
    for (std::size_t b = 0; b < fast_bytes; ++b) {
        std::uint64_t word;
        std::memcpy(&word, &seq.data[b], 8);
        if constexpr (Order == BitOrder::Msb) {
            // Msb needs a byte swap to match the bit order on little endian systems: We are
            // reading 8 bytes as a 64-bit word, from a stream that consists of individual bytes.
            word = byte_swap_64(word);
        }

        // Barrier every k-mer of this byte before consuming any of them (see file-level comment
        // above on do_not_optimize).
        auto const v0 = extract_(word, 0);
        auto const v1 = extract_(word, 1);
        auto const v2 = extract_(word, 2);
        auto const v3 = extract_(word, 3);
        do_not_optimize(v0);
        do_not_optimize(v1);
        do_not_optimize(v2);
        do_not_optimize(v3);
        func(v0);
        func(v1);
        func(v2);
        func(v3);
    }

    // See for_each_kmer_packed_narrow_tail_()'s own docs above for why this stays out of line,
    // and why `func` is only ever called from here, never handed to that helper directly.
    std::array<std::uint64_t, 32> tail_vals;
    std::size_t const tail_n = for_each_kmer_packed_narrow_tail_<Order>(
        seq, fast_bytes, num_bytes, p_max, k32, mask, tail_vals
    );
    for (std::size_t i = 0; i < tail_n; ++i) {
        func(tail_vals[i]);
    }
}

// -------------------------------------------------------------------------------------------------
//     Compile-time K
// -------------------------------------------------------------------------------------------------

template <BitOrder Order, unsigned K, typename Func>
inline void for_each_kmer_packed_narrow_fixed_k_impl_(TwoBitSequence<Order> const& seq, Func& func)
{
    static_assert(K >= 1 && K <= 29, "K must be in [1, 29]");

    std::uint64_t const mask = (std::uint64_t{1} << (2 * K)) - 1u;

    std::size_t const p_max = seq.length - K;
    std::size_t const full_bytes = (p_max + 1) / 4;
    std::size_t const num_bytes  = (seq.length + 3) / 4;
    std::size_t const fast_bytes = std::min(
        full_bytes, num_bytes >= 8 ? num_bytes - 8 : std::size_t{0}
    );

    auto extract_ = [&](std::uint64_t word, unsigned local) -> std::uint64_t {
        if constexpr (Order == BitOrder::Msb) {
            return (word >> (64 - 2 * local - 2 * K)) & mask;
        } else {
            return (word >> (2 * local)) & mask;
        }
    };

    for (std::size_t b = 0; b < fast_bytes; ++b) {
        std::uint64_t word;
        std::memcpy(&word, &seq.data[b], 8);
        if constexpr (Order == BitOrder::Msb) {
            word = byte_swap_64(word);
        }
        auto const v0 = extract_(word, 0);
        auto const v1 = extract_(word, 1);
        auto const v2 = extract_(word, 2);
        auto const v3 = extract_(word, 3);

        do_not_optimize(v0);
        do_not_optimize(v1);
        do_not_optimize(v2);
        do_not_optimize(v3);

        func(v0);
        func(v1);
        func(v2);
        func(v3);
    }

    // Reuses the existing runtime-k tail helper unchanged: the tail handles at most 32 k-mers
    // total, so it's not worth a separate K-templated copy purely for that small a hot path.
    std::array<std::uint64_t, 32> tail_vals;
    std::size_t const tail_n = for_each_kmer_packed_narrow_tail_<Order>(
        seq, fast_bytes, num_bytes, p_max, K, mask, tail_vals
    );
    for (std::size_t i = 0; i < tail_n; ++i) {
        func(tail_vals[i]);
    }
}

template <BitOrder Order, typename FuncD, std::size_t... Ks>
inline auto for_each_kmer_packed_narrow_fixed_k_table_(std::index_sequence<Ks...>)
{
    using FnPtr = void (*)(TwoBitSequence<Order> const&, FuncD&);
    return std::array<FnPtr, sizeof...(Ks)>{
        &for_each_kmer_packed_narrow_fixed_k_impl_<Order, static_cast<unsigned>(Ks) + 1, FuncD>...
    };
}

/**
 * @brief Extract all k-mers for k in [1, 29] directly from a packed TwoBitSequence, and call a
 * callback on each, with a compile-time-resolved k.
 *
 * This variation allows the compiler to optimize away some runtime overhead, which can be
 * more performant in some cases, at the cost of code size (29 separate instances of the main loop).
 */
template <BitOrder Order, typename Func>
inline void for_each_kmer_packed_narrow_fixed_k(
    TwoBitSequence<Order> const& seq, std::size_t k, Func&& func
) {
    if (k == 0 || k > 29) {
        throw_invalid_kmer_k_(29);
    }
    if (seq.length < k) {
        return;
    }

    using FuncD = std::remove_reference_t<Func>;
    static auto const table = for_each_kmer_packed_narrow_fixed_k_table_<Order, FuncD>(
        std::make_index_sequence<29>{}
    );
    table[k - 1](seq, func);
}

// -------------------------------------------------------------------------------------------------
//     Rolling
// -------------------------------------------------------------------------------------------------

/**
 * @brief Extract all k-mers for k in [1, 29] from a packed TwoBitSequence via a single rolling
 * 64-bit accumulator, folding one byte at a time. Kept only as a slower algorithmic comparison
 * baseline; prefer the other methods for production use.
 */
template <BitOrder Order, typename Func>
inline void for_each_kmer_packed_narrow_rolling(
    TwoBitSequence<Order> const& seq, std::size_t k, Func&& func
) {
    if (k == 0 || k > 29) {
        throw_invalid_kmer_k_(29);
    }
    if (seq.length < k) {
        return;
    }

    std::uint64_t const mask = (std::uint64_t{1} << (2 * k)) - 1u;

    std::uint64_t acc = 0;

    // Incrementally folds one byte at a time into `acc`, each byte's fold depending serially on
    // the previous one -- as opposed to for_each_kmer_packed_narrow_blockwise()'s independent,
    // overlapping per-byte word loads. That serial dependency chain is what makes this version
    // slower.
    auto fold_byte_ = [&](std::uint8_t byte) {
        if constexpr (Order == BitOrder::Msb) {
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
            if constexpr (Order == BitOrder::Msb) {
                shift = s;
            } else {
                shift = 64 - 2 * static_cast<unsigned>(k) - s;
            }
            func((acc >> shift) & mask);
        }
    }
}

// =================================================================================================
//     Wide (1 <= k <= 32) K-mer Extraction
// =================================================================================================

// -------------------------------------------------------------------------------------------------
//     Helpers
// -------------------------------------------------------------------------------------------------

// Shared tail for every variant in this section, decoding one base at a time (O(k) per k-mer)
// rather than reusing a main loop's word-width trick -- simple and obviously correct, and only
// ever runs for the handful of k-mers a variant's fast path excludes. As in
// for_each_kmer_packed_narrow_tail_() above, `func` is never passed in directly (same reason).
template <BitOrder Order>
[[gnu::noinline, gnu::cold]]
std::size_t for_each_kmer_packed_wide_tail_(
    TwoBitSequence<Order> const& seq, std::size_t start_pos, std::size_t p_max,
    unsigned k32, std::array<std::uint64_t, 64>& out
) {
    auto decode_base_ = [&](std::size_t p) -> unsigned {
        std::uint8_t const byte = seq.data[p / 4];
        unsigned const in_byte = static_cast<unsigned>(p % 4);
        if constexpr (Order == BitOrder::Msb) {
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
            if constexpr (Order == BitOrder::Msb) {
                kmer = (kmer << 2) | code;
            } else {
                kmer |= std::uint64_t{code} << (2 * i);
            }
        }
        out[n++] = kmer;
    }
    return n;
}

// -------------------------------------------------------------------------------------------------
//     Blockwise
// -------------------------------------------------------------------------------------------------

/**
 * @brief Extract all k-mers for k in [1, 32] directly from a packed TwoBitSequence, and call a
 * callback on each, using blockwise processing of the sequence.
 *
 * Compared to for_each_kmer_packed_narrow_blockwise(), this variant reads an extra byte per byte
 * position, to cover k up to 32, at a slight expense in performance.
 */
template <BitOrder Order, typename Func>
inline void for_each_kmer_packed_wide_blockwise(
    TwoBitSequence<Order> const& seq, std::size_t k, Func&& func
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

    // Same reasoning as for_each_kmer_packed_narrow_blockwise() fast_bytes (see above),
    // but this function's window_() needs 9 bytes (8 + 1), not 8, to cover k up to 32.
    std::size_t const fast_bytes =
        (num_bytes >= 9) ? std::min(full_bytes, num_bytes - 9) : std::size_t{0}
    ;

    // Bits [start + 2k - 1 : start] of the combined 128-bit (hi:lo) value, right-justified.
    auto window_ = [&](std::uint64_t hi, std::uint64_t lo, unsigned local) -> std::uint64_t {
        unsigned start;
        if constexpr (Order == BitOrder::Msb) {
            start = 128 - 2 * local - 2 * k32;
        } else {
            start = 2 * local;
        }
        std::uint64_t bits;
        if (start >= 64) {
            bits = hi >> (start - 64);
        } else if (start + 2 * k32 <= 64) {
            bits = lo >> start;
        } else {
            bits = (lo >> start) | (hi << (64 - start));
        }
        return bits & mask;
    };

    for (std::size_t b = 0; b < fast_bytes; ++b) {
        std::uint64_t lo;
        std::memcpy(&lo, &seq.data[b], 8);
        std::uint64_t hi = static_cast<std::uint64_t>(seq.data[b + 8]);
        if constexpr (Order == BitOrder::Msb) {
            // Mirrors for_each_kmer_packed_narrow_blockwise() single byte_swap_64(), extended:
            // `hi`'s one real byte moves to lo's top byte (position 8 of the 9), and the swapped
            // 8-byte word becomes `hi` (positions 0-7), together placing byte 0 (data[b]) at
            // the very top of the combined 72-bit value, as this function's Msb convention requires.
            std::uint64_t const swapped_lo = byte_swap_64(lo);
            lo = hi << 56;
            hi = swapped_lo;
        }

        auto const v0 = window_(hi, lo, 0);
        auto const v1 = window_(hi, lo, 1);
        auto const v2 = window_(hi, lo, 2);
        auto const v3 = window_(hi, lo, 3);

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
    std::size_t const tail_n = for_each_kmer_packed_wide_tail_<Order>(
        seq, 4 * fast_bytes, p_max, k32, tail_vals
    );
    for (std::size_t i = 0; i < tail_n; ++i) {
        func(tail_vals[i]);
    }
}

// -------------------------------------------------------------------------------------------------
//     Compile-time K
// -------------------------------------------------------------------------------------------------

// Same 8+1-byte independent read as for_each_kmer_packed_wide_blockwise(), but with `k` itself a
// template parameter: every `start` and case-selection decision window_() computes becomes a
// compile-time constant, foldable via `if constexpr`, instead of merely loop-invariant.
// for_each_kmer_packed_wide_fixed_k_impl_() is the per-K implementation;
// for_each_kmer_packed_wide_fixed_k() dispatches a runtime `k` to the matching instantiation via a
// once-built function-pointer table.
template <BitOrder Order, unsigned K>
inline std::uint64_t wide_fixed_k_window_(
    std::uint64_t hi, std::uint64_t lo, unsigned local, std::uint64_t mask
) {
    // local is a runtime loop-unrolled argument (0..3) here rather than its own template
    // parameter purely to keep the 4 call sites below uniform with this family's other
    // variants; since K is compile-time and this is called with a literal `local` at each site,
    // the compiler already constant-folds `start` itself just as fully either way.
    unsigned start;
    if constexpr (Order == BitOrder::Msb) {
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

template <BitOrder Order, unsigned K, typename Func>
inline void for_each_kmer_packed_wide_fixed_k_impl_(TwoBitSequence<Order> const& seq, Func& func)
{
    static_assert(K >= 1 && K <= 32, "K must be in [1, 32]");

    std::uint64_t const mask = (K == 32) ? ~std::uint64_t{0} : ((std::uint64_t{1} << (2 * K)) - 1u);

    std::size_t const p_max = seq.length - K;
    std::size_t const full_bytes = (p_max + 1) / 4;
    std::size_t const num_bytes  = (seq.length + 3) / 4;

    std::size_t const fast_bytes = (num_bytes >= 9) ? std::min(full_bytes, num_bytes - 9)
                                                     : std::size_t{0};

    for (std::size_t b = 0; b < fast_bytes; ++b) {
        std::uint64_t lo;
        std::memcpy(&lo, &seq.data[b], 8);
        std::uint64_t hi = static_cast<std::uint64_t>(seq.data[b + 8]);
        if constexpr (Order == BitOrder::Msb) {
            std::uint64_t const swapped_lo = byte_swap_64(lo);
            lo = hi << 56;
            hi = swapped_lo;
        }

        auto const v0 = wide_fixed_k_window_<Order, K>(hi, lo, 0, mask);
        auto const v1 = wide_fixed_k_window_<Order, K>(hi, lo, 1, mask);
        auto const v2 = wide_fixed_k_window_<Order, K>(hi, lo, 2, mask);
        auto const v3 = wide_fixed_k_window_<Order, K>(hi, lo, 3, mask);

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
    std::size_t const tail_n = for_each_kmer_packed_wide_tail_<Order>(
        seq, 4 * fast_bytes, p_max, K, tail_vals
    );
    for (std::size_t i = 0; i < tail_n; ++i) {
        func(tail_vals[i]);
    }
}

template <BitOrder Order, typename FuncD, std::size_t... Ks>
inline auto for_each_kmer_packed_wide_fixed_k_table_(std::index_sequence<Ks...>)
{
    using FnPtr = void (*)(TwoBitSequence<Order> const&, FuncD&);
    return std::array<FnPtr, sizeof...(Ks)>{
        &for_each_kmer_packed_wide_fixed_k_impl_<Order, static_cast<unsigned>(Ks) + 1, FuncD>...
    };
}

/**
 * @brief Extract all k-mers for k in [1, 32] directly from a packed TwoBitSequence, and call a
 * callback on each, with a compile-time-resolved k.
 *
 * This variation allows the compiler to optimize away some runtime overhead, which can be
 * more performant in some cases, at the cost of code size (32 separate instances of the main loop).
 */
template <BitOrder Order, typename Func>
inline void for_each_kmer_packed_wide_fixed_k(
    TwoBitSequence<Order> const& seq, std::size_t k, Func&& func
) {
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.length < k) {
        return;
    }

    using FuncD = std::remove_reference_t<Func>;
    static auto const table = for_each_kmer_packed_wide_fixed_k_table_<Order, FuncD>(
        std::make_index_sequence<32>{}
    );
    table[k - 1](seq, func);
}

// -------------------------------------------------------------------------------------------------
//     Rolling
// -------------------------------------------------------------------------------------------------

/**
 * @brief Extract all k-mers for k in [1, 32] from a packed TwoBitSequence via a rolling hi:lo
 * accumulator, folding one byte at a time.
 *
 * This is the wide-k-mer analog of for_each_kmer_packed_narrow_rolling(). Kept only as a slower
 * algorithmic comparison baseline; prefer other variants for production use.
 */
template <BitOrder Order, typename Func>
inline void for_each_kmer_packed_wide_rolling(
    TwoBitSequence<Order> const& seq, std::size_t k, Func&& func
) {
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.length < k) {
        return;
    }

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
        if constexpr (Order == BitOrder::Msb) {
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
        if constexpr (Order == BitOrder::Msb) {
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

// -------------------------------------------------------------------------------------------------
//     128-bit
// -------------------------------------------------------------------------------------------------

// `unsigned __int128` is a GCC/Clang extension, not ISO C++, so both guard its use behind
// `__SIZEOF_INT128__` (defined by the compiler exactly when a 128-bit integer type exists, so
// for_each_kmer_packed_wide_128() and its callers simply don't get compiled on targets without
// one, e.g. 32-bit or MSVC) and locally silence `-Wpedantic` around the one function that uses
// it, rather than disabling that warning project-wide for the sake of this single variant.
#ifdef __SIZEOF_INT128__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"

/**
 * @brief Extract all k-mers for k in [1, 32] directly from a packed TwoBitSequence, and call a
 * callback on each, by reading 16 bytes per byte position as a single native 128-bit load
 * (`unsigned __int128`).
 *
 * Exploratory only -- not recommended for production use; kept as a comparison point against the
 * hi:lo-split variants below.
 */
template <BitOrder Order, typename Func>
inline void for_each_kmer_packed_wide_128(
    TwoBitSequence<Order> const& seq, std::size_t k, Func&& func
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

    std::size_t const fast_bytes = (num_bytes >= 16) ? std::min(full_bytes, num_bytes - 16)
                                                      : std::size_t{0};

    // Same bits-and-case logic as the hi:lo-split variants' window_(), just directly on a 128-bit
    // value instead of a manually split pair.
    auto window_ = [&](unsigned __int128 word, unsigned local) -> std::uint64_t {
        unsigned start;
        if constexpr (Order == BitOrder::Msb) {
            start = 128 - 2 * local - 2 * k32;
        } else {
            start = 2 * local;
        }
        return static_cast<std::uint64_t>(word >> start) & mask;
    };

    for (std::size_t b = 0; b < fast_bytes; ++b) {
        unsigned __int128 word;
        std::memcpy(&word, &seq.data[b], 16);
        if constexpr (Order == BitOrder::Msb) {
            std::uint64_t const lo64 = static_cast<std::uint64_t>(word);
            std::uint64_t const hi64 = static_cast<std::uint64_t>(word >> 64);
            word = (static_cast<unsigned __int128>(byte_swap_64(lo64)) << 64) | byte_swap_64(hi64);
        }

        auto const v0 = window_(word, 0);
        auto const v1 = window_(word, 1);
        auto const v2 = window_(word, 2);
        auto const v3 = window_(word, 3);

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
    std::size_t const tail_n = for_each_kmer_packed_wide_tail_<Order>(
        seq, 4 * fast_bytes, p_max, k32, tail_vals
    );
    for (std::size_t i = 0; i < tail_n; ++i) {
        func(tail_vals[i]);
    }
}

#pragma GCC diagnostic pop
#endif // __SIZEOF_INT128__

// -------------------------------------------------------------------------------------------------
//     Hybrid
// -------------------------------------------------------------------------------------------------

/**
 * @brief Same as for_each_kmer_packed_wide_blockwise(), but window_() picks its extraction
 * strategy per `Order`: the branching formula for Msb, an unconditional branch-free formula for
 * Lsb.
 */
template <BitOrder Order, typename Func>
inline void for_each_kmer_packed_wide_hybrid(
    TwoBitSequence<Order> const& seq, std::size_t k, Func&& func
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

    // `Order` is a template parameter, so each instantiation compiles only the branch it takes:
    // Msb's `start` depends on runtime `k`, so the branch-free formula's extra masking can't fold
    // away there and the branching form is cheaper; Lsb's `start` is a compile-time constant per
    // unrolled `local`, so the branch-free form folds down to just the useful shifts instead.
    auto window_ = [&](std::uint64_t hi, std::uint64_t lo, unsigned local) -> std::uint64_t {
        if constexpr (Order == BitOrder::Msb) {
            unsigned const start = 128 - 2 * local - 2 * k32;
            std::uint64_t bits;
            if (start >= 64) {
                bits = hi >> (start - 64);
            } else if (start + 2 * k32 <= 64) {
                bits = lo >> start;
            } else {
                bits = (lo >> start) | (hi << (64 - start));
            }
            return bits & mask;
        } else {
            // Unconditional (branch-free) form of the same bits-and-case semantics as the Msb
            // branch above. `s` is the shift always safe to apply directly (0..63); `sel` (all-1s
            // or all-0s, from a plain comparison, not a branch) swaps in `hi` as the "low" half
            // once start>=64, matching Msb's hi-only case. The hi contribution itself is split
            // into two safe sub-64 shifts (`<<1` then `<<(63-s)`) rather than a single `<<(64-s)`,
            // since that single shift is undefined right at s=0 -- the two-step version instead
            // correctly evaluates to 0 there, which is what's needed for the lo-only case (any
            // nonzero bits it produces elsewhere land above bit 2k-1 and are discarded by the
            // final mask below anyway, same as the spans case relies on).
            unsigned const start = 2 * local;
            unsigned const s = start & 63u;
            std::uint64_t const sel = 0u - static_cast<std::uint64_t>(start >= 64);
            std::uint64_t const lo_sel = (lo & ~sel) | (hi & sel);
            std::uint64_t const hi_sel = hi & ~sel;
            std::uint64_t const bits = (lo_sel >> s) | ((hi_sel << 1) << (63 - s));
            return bits & mask;
        }
    };

    for (std::size_t b = 0; b < fast_bytes; ++b) {
        std::uint64_t lo;
        std::memcpy(&lo, &seq.data[b], 8);
        std::uint64_t hi = static_cast<std::uint64_t>(seq.data[b + 8]);
        if constexpr (Order == BitOrder::Msb) {
            std::uint64_t const swapped_lo = byte_swap_64(lo);
            lo = hi << 56;
            hi = swapped_lo;
        }

        auto const v0 = window_(hi, lo, 0);
        auto const v1 = window_(hi, lo, 1);
        auto const v2 = window_(hi, lo, 2);
        auto const v3 = window_(hi, lo, 3);

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
    std::size_t const tail_n = for_each_kmer_packed_wide_tail_<Order>(
        seq, 4 * fast_bytes, p_max, k32, tail_vals
    );
    for (std::size_t i = 0; i < tail_n; ++i) {
        func(tail_vals[i]);
    }
}

// -------------------------------------------------------------------------------------------------
//     Hoisted
// -------------------------------------------------------------------------------------------------

/**
 * @brief Same as for_each_kmer_packed_wide_blockwise(), but with window_()'s per-local `start`
 * and case selection precomputed once before the main loop instead of re-derived on every call.
 */
template <BitOrder Order, typename Func>
inline void for_each_kmer_packed_wide_hoisted(
    TwoBitSequence<Order> const& seq, std::size_t k, Func&& func
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

    // Per-local extraction plan: `start` and which of window_()'s 3 cases it falls into, same as
    // for_each_kmer_packed_wide_rolling()'s own LocalPlan. Both depend only on `k` and
    // `local`, never on the byte data, so precomputing them here avoids re-deriving this 4 times
    // per byte for the whole sequence.
    struct LocalPlan
    {
        unsigned start;
        int case_id; // 0: hi only, 1: lo only, 2: spans hi and lo
    };
    std::array<LocalPlan, 4> plans;
    for (unsigned local = 0; local < 4; ++local) {
        unsigned start;
        if constexpr (Order == BitOrder::Msb) {
            start = 128 - 2 * local - 2 * k32;
        } else {
            start = 2 * local;
        }
        int case_id;
        if (start >= 64) {
            case_id = 0;
        } else if (start + 2 * k32 <= 64) {
            case_id = 1;
        } else {
            case_id = 2;
        }
        plans[local] = LocalPlan{start, case_id};
    }

    auto window_ = [&](std::uint64_t hi, std::uint64_t lo, LocalPlan const& p) -> std::uint64_t {
        std::uint64_t bits;
        switch (p.case_id) {
            case 0:  bits = hi >> (p.start - 64); break;
            case 1:  bits = lo >> p.start; break;
            default: bits = (lo >> p.start) | (hi << (64 - p.start)); break;
        }
        return bits & mask;
    };

    for (std::size_t b = 0; b < fast_bytes; ++b) {
        std::uint64_t lo;
        std::memcpy(&lo, &seq.data[b], 8);
        std::uint64_t hi = static_cast<std::uint64_t>(seq.data[b + 8]);
        if constexpr (Order == BitOrder::Msb) {
            std::uint64_t const swapped_lo = byte_swap_64(lo);
            lo = hi << 56;
            hi = swapped_lo;
        }

        auto const v0 = window_(hi, lo, plans[0]);
        auto const v1 = window_(hi, lo, plans[1]);
        auto const v2 = window_(hi, lo, plans[2]);
        auto const v3 = window_(hi, lo, plans[3]);

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
    std::size_t const tail_n = for_each_kmer_packed_wide_tail_<Order>(
        seq, 4 * fast_bytes, p_max, k32, tail_vals
    );
    for (std::size_t i = 0; i < tail_n; ++i) {
        func(tail_vals[i]);
    }
}

// -------------------------------------------------------------------------------------------------
//     Hybrid + Hoisted
// -------------------------------------------------------------------------------------------------

/**
 * @brief Combines for_each_kmer_packed_wide_hybrid()'s per-`Order` formula choice with
 * for_each_kmer_packed_wide_hoisted()'s once-per-call precomputed local plan, applying each only
 * where it helps: Lsb's branch-free formula needs no hoisting (its shift/select values depend
 * only on `local`, already compile-time-constant regardless of runtime `k`), so only Msb's
 * case-selection plan is precomputed here.
 *
 * Does not consistently outperform either for_each_kmer_packed_wide_hybrid() or
 * for_each_kmer_packed_wide_hoisted() alone; kept as a comparison point, not recommended over them.
 */
template <BitOrder Order, typename Func>
inline void for_each_kmer_packed_wide_hybrid_hoisted(
    TwoBitSequence<Order> const& seq, std::size_t k, Func&& func
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

    // Msb-only plan (see this function's own docs for why Lsb needs none): start/case-selection
    // depends on runtime k, so hoist it once per local instead of re-deriving 4 times per byte.
    struct LocalPlan
    {
        unsigned start;
        int case_id; // 0: hi only, 1: lo only, 2: spans hi and lo
    };
    std::array<LocalPlan, 4> plans{};
    if constexpr (Order == BitOrder::Msb) {
        for (unsigned local = 0; local < 4; ++local) {
            unsigned const start = 128 - 2 * local - 2 * k32;
            int case_id;
            if (start >= 64) {
                case_id = 0;
            } else if (start + 2 * k32 <= 64) {
                case_id = 1;
            } else {
                case_id = 2;
            }
            plans[local] = LocalPlan{start, case_id};
        }
    }

    auto window_ = [&](std::uint64_t hi, std::uint64_t lo, unsigned local) -> std::uint64_t {
        if constexpr (Order == BitOrder::Msb) {
            LocalPlan const& p = plans[local];
            std::uint64_t bits;
            switch (p.case_id) {
                case 0:  bits = hi >> (p.start - 64); break;
                case 1:  bits = lo >> p.start; break;
                default: bits = (lo >> p.start) | (hi << (64 - p.start)); break;
            }
            return bits & mask;
        } else {
            // Branch-free form, same as for_each_kmer_packed_wide_hybrid()'s Lsb branch -- see
            // its own docs for why the two-step `hi` shift is needed at s=0.
            unsigned const start = 2 * local;
            unsigned const s = start & 63u;
            std::uint64_t const sel = 0u - static_cast<std::uint64_t>(start >= 64);
            std::uint64_t const lo_sel = (lo & ~sel) | (hi & sel);
            std::uint64_t const hi_sel = hi & ~sel;
            std::uint64_t const bits = (lo_sel >> s) | ((hi_sel << 1) << (63 - s));
            return bits & mask;
        }
    };

    for (std::size_t b = 0; b < fast_bytes; ++b) {
        std::uint64_t lo;
        std::memcpy(&lo, &seq.data[b], 8);
        std::uint64_t hi = static_cast<std::uint64_t>(seq.data[b + 8]);
        if constexpr (Order == BitOrder::Msb) {
            std::uint64_t const swapped_lo = byte_swap_64(lo);
            lo = hi << 56;
            hi = swapped_lo;
        }

        auto const v0 = window_(hi, lo, 0);
        auto const v1 = window_(hi, lo, 1);
        auto const v2 = window_(hi, lo, 2);
        auto const v3 = window_(hi, lo, 3);

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
    std::size_t const tail_n = for_each_kmer_packed_wide_tail_<Order>(
        seq, 4 * fast_bytes, p_max, k32, tail_vals
    );
    for (std::size_t i = 0; i < tail_n; ++i) {
        func(tail_vals[i]);
    }
}

// =================================================================================================
//     Aligned Windows (Experimental)
// =================================================================================================

/**
 * @brief Narrow extraction with one shared runtime alignment for Msb, then constant local shifts.
 * Lsb already uses constant shifts, so its arithmetic is unchanged from narrow_blockwise.
 */
template <BitOrder Order, typename Func>
inline void for_each_kmer_packed_narrow_aligned(
    TwoBitSequence<Order> const& seq, std::size_t k, Func&& func
) {
    if (k == 0 || k > 29) {
        throw_invalid_kmer_k_(29);
    }
    if (seq.length < k) {
        return;
    }

    unsigned const k32 = static_cast<unsigned>(k);
    std::uint64_t const mask = (std::uint64_t{1} << (2 * k32)) - 1u;
    std::size_t const p_max = seq.length - k;
    std::size_t const full_bytes = (p_max + 1) / 4;
    std::size_t const num_bytes = (seq.length + 3) / 4;
    // Match the existing variants' conservative read margin to isolate the arithmetic change.
    std::size_t const fast_bytes = std::min(
        full_bytes, num_bytes >= 8 ? num_bytes - 8 : std::size_t{0}
    );

    for (std::size_t b = 0; b < fast_bytes; ++b) {
        std::uint64_t word;
        std::memcpy(&word, &seq.data[b], 8);
        if constexpr (Order == BitOrder::Msb) {
            // 58-2k is in [0,56]; adding the local shift recovers 64-2k-2*local.
            word = byte_swap_64(word) >> (58 - 2 * k32);
        }
        std::uint64_t v0, v1, v2, v3;
        if constexpr (Order == BitOrder::Msb) {
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

        // Preserve the existing materialization and callback order in every experimental loop.
        do_not_optimize(v0);
        do_not_optimize(v1);
        do_not_optimize(v2);
        do_not_optimize(v3);
        func(v0);
        func(v1);
        func(v2);
        func(v3);
    }

    std::array<std::uint64_t, 32> tail_vals;
    std::size_t const tail_n = for_each_kmer_packed_narrow_tail_<Order>(
        seq, fast_bytes, num_bytes, p_max, k32, mask, tail_vals
    );
    for (std::size_t i = 0; i < tail_n; ++i) {
        func(tail_vals[i]);
    }
}

// Build a 32-base window first; selecting k then needs no boundary-dependent case selection.
// Local is a template argument so the Lsb local=0 case never forms a shift by 64.
template <BitOrder Order, unsigned Local>
inline std::uint64_t packed_aligned_window_(
    std::uint64_t word, std::uint64_t extra, unsigned right_shift, std::uint64_t mask
) {
    static_assert(Local < 4, "Local must be in [0, 3]");
    unsigned constexpr s = 2 * Local;
    if constexpr (Order == BitOrder::Msb) {
        // s is [0,6], 8-s is [2,8], and right_shift is [0,62]. Unsigned left shifts
        // intentionally discard the bases before Local; extra supplies the missing low bits.
        return ((word << s) | (extra >> (8 - s))) >> right_shift;
    } else if constexpr (Local == 0) {
        return word & mask;
    } else {
        // s is [2,6], so 64-s is [58,62]; extra is already widened to uint64_t.
        return ((word >> s) | (extra << (64 - s))) & mask;
    }
}

// Shared body for runtime-k wide_aligned (K=0) and wide_split_k's three wide specializations.
// Callers validate k and sequence length before entering; no callback type erasure is needed.
template <BitOrder Order, unsigned K, typename Func>
inline void for_each_kmer_packed_wide_aligned_impl_(
    TwoBitSequence<Order> const& seq, std::size_t runtime_k, Func& func
) {
    static_assert(K == 0 || (K >= 30 && K <= 32), "K must be 0 (runtime) or in [30, 32]");
    unsigned const k32 = K == 0 ? static_cast<unsigned>(runtime_k) : K;
    std::uint64_t const mask = k32 == 32 ? ~std::uint64_t{0}
        : (std::uint64_t{1} << (2 * k32)) - 1u;
    unsigned const right_shift = 64 - 2 * k32;
    std::size_t const p_max = seq.length - k32;
    std::size_t const full_bytes = (p_max + 1) / 4;
    std::size_t const num_bytes = (seq.length + 3) / 4;
    // Keep the same 8+1-byte loads and tail cutoff as wide_blockwise.
    std::size_t const fast_bytes = num_bytes >= 9
        ? std::min(full_bytes, num_bytes - 9) : std::size_t{0};

    for (std::size_t b = 0; b < fast_bytes; ++b) {
        std::uint64_t word;
        std::memcpy(&word, &seq.data[b], 8);
        std::uint64_t const extra = seq.data[b + 8];
        if constexpr (Order == BitOrder::Msb) {
            word = byte_swap_64(word);
        }
        auto const v0 = packed_aligned_window_<Order, 0>(word, extra, right_shift, mask);
        auto const v1 = packed_aligned_window_<Order, 1>(word, extra, right_shift, mask);
        auto const v2 = packed_aligned_window_<Order, 2>(word, extra, right_shift, mask);
        auto const v3 = packed_aligned_window_<Order, 3>(word, extra, right_shift, mask);

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
    std::size_t const tail_n = for_each_kmer_packed_wide_tail_<Order>(
        seq, 4 * fast_bytes, p_max, k32, tail_vals
    );
    for (std::size_t i = 0; i < tail_n; ++i) {
        func(tail_vals[i]);
    }
}

/** @brief Runtime-k extraction for k in [1,32], using constant-offset 64-bit windows. */
template <BitOrder Order, typename Func>
inline void for_each_kmer_packed_wide_aligned(
    TwoBitSequence<Order> const& seq, std::size_t k, Func&& func
) {
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.length < k) {
        return;
    }
    for_each_kmer_packed_wide_aligned_impl_<Order, 0>(seq, k, func);
}

/** @brief Use narrow_aligned for k<=29 and direct wide specializations for k=30,31,32. */
template <BitOrder Order, typename Func>
inline void for_each_kmer_packed_wide_split_k(
    TwoBitSequence<Order> const& seq, std::size_t k, Func&& func
) {
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.length < k) {
        return;
    }
    // Dispatch once per sequence, with three direct calls instead of a 32-entry pointer table.
    switch (k) {
        case 30: for_each_kmer_packed_wide_aligned_impl_<Order, 30>(seq, k, func); break;
        case 31: for_each_kmer_packed_wide_aligned_impl_<Order, 31>(seq, k, func); break;
        case 32: for_each_kmer_packed_wide_aligned_impl_<Order, 32>(seq, k, func); break;
        default: for_each_kmer_packed_narrow_aligned(seq, k, func); break;
    }
}
