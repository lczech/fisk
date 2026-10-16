#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/kmer_extract.hpp"

namespace fisk {

// Both MSB and LSB below load larger words from raw bytes via memcpy and rely on how the host
// interprets them as an integer. Both are therefore little-endian-specific; see
// core/platform.hpp for the assertion that enforces this.

// =================================================================================================
//     K-mer Extraction from a PackedSequence
// =================================================================================================

// -------------------------------------------------------------------------------------------------
//     Overview
// -------------------------------------------------------------------------------------------------

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
// Both conventions are handed on to the callback, which receives a Kmer<E, L> (core/kmer.hpp)
// rather than a bare word, so a k-mer cannot be read under conventions other than those it was
// produced with. The loops themselves compute in raw std::uint64_t throughout and only adopt the
// type at the point of emission, so that tagging costs no instructions.
//
// Two public functions cover the full documented k in [1, 32]: for_each_kmer_packed_aligned()
// (the fast path) and for_each_kmer_packed_rolling() (a simpler reference baseline, not
// recommended for production). Both internally specialize for k <= 29 (where a single
// 64-bit accumulator suffices) vs k > 29 (needing an extra byte for spilling).
//
// Their checked counterparts, for_each_kmer_packed_checked_aligned() and
// for_each_kmer_packed_checked_rolling(), take a PackedSequenceWithValidity instead, and emit only
// the k-mers whose nucleotides are all valid by its validity bits.

// -------------------------------------------------------------------------------------------------
//     Details
// -------------------------------------------------------------------------------------------------

// Precondition-check throws are routed through the shared throw_invalid_kmer_k_()
// (kmer_extract.hpp) rather than thrown directly, trying to keep these functions small enough for
// the compiler to treat as inlining candidates.

// do_not_optimize barriers on the k-mer emission in some of the functions below force each value
// to at least be manifested in a register, so a benchmark summing them cannot be proven reducible
// by the compiler. They emit no instructions on GCC/Clang, but their register requirements might
// still slightly affect regular callers -- so they only compile in when FISK_BENCHMARK_SINK_MODE
// is defined (see benchmarks/sink.hpp and CMakeLists.txt's fisk_benchmarks target), i.e. only for
// the benchmark suite itself, never for regular library consumers. For benchmarking, they _need_
// to stay inside those functions, so that the compiler can reorder instructions within the
// function. If we were to add those barriers in the benchmark itself, the compiler would be
// forced to serialize the output, severely messing with the scheduling and benchmarks.

// We experimented with several variations of the basic algorithm here, in order to find a solution
// that compiles to optimal code across most compilers and platforms. Some of the more promising
// variants are in commits
//  -f3910762f1d4a577da88de02347faf94e584a0f1
//  -313e33b54b07e87ec3a4852c65403d9ed77050b6
// but we since have cleaned up a bit, and are only keeping a final selection of variants here.

// -------------------------------------------------------------------------------------------------
//     Helpers
// -------------------------------------------------------------------------------------------------

// The k-mers after the fast region of a loop, of which there are at most 56, for any length of the
// sequence: the region ends at most 40 positions before the end, as its 8-byte loads have to be
// within the codes, and at most 56 for the checked loops, see checked_fast_bytes_().
using PackedTail_ = std::array<std::uint64_t, 64>;

// Shared tail for every fast-path variant below, decoding one base at a time (O(k) per k-mer)
// rather than reusing a main loop's word-width trick; only ever runs for the handful of k-mers
// after the fast path. The callback function is never passed in directly, as that would prevent
// compiler optimizations; instead, values are returned to the caller.
template <Encoding E, Layout L>
FISK_NOINLINE_COLD
std::size_t collect_kmers_packed_tail_(
    PackedSequence<E, L> const& seq, std::size_t start_pos, std::size_t p_max,
    unsigned k32, PackedTail_& out
) {
    auto decode_base_ = [&](std::size_t p) -> unsigned {
        std::uint8_t const byte = seq.data[p / 4];
        unsigned const in_byte = static_cast<unsigned>(p % 4);
        if constexpr (L == Layout::kMSB) {
            return (byte >> (6 - 2 * in_byte)) & 0x3u;
        } else if constexpr (L == Layout::kLSB) {
            return (byte >> (2 * in_byte)) & 0x3u;
        } else {
            static_assert(dependent_false_v<L>, "Unhandled Layout in collect_kmers_packed_tail_()");
        }
    };

    assert(p_max + 1 <= start_pos + out.size());
    std::size_t n = 0;
    for (std::size_t p = start_pos; p <= p_max; ++p) {
        std::uint64_t kmer = 0;
        for (unsigned i = 0; i < k32; ++i) {
            unsigned const code = decode_base_(p + i);
            if constexpr (L == Layout::kMSB) {
                kmer = (kmer << 2) | code;
            } else if constexpr (L == Layout::kLSB) {
                kmer |= std::uint64_t{code} << (2 * i);
            } else {
                static_assert(
                    dependent_false_v<L>, "Unhandled Layout in collect_kmers_packed_tail_()"
                );
            }
        }
        out[n++] = kmer;
    }
    return n;
}

// -------------------------------------------------------------------------------------------------
//     Validity
// -------------------------------------------------------------------------------------------------

// The checked functions read the validity bits of a PackedSequenceWithValidity, see core/types.hpp,
// as windows of 64 nucleotides: bit `j` of the window at `pos` is set if nucleotide `pos + j` is
// valid, and a k-mer at `pos` is valid if bits 0 to k - 1 of its window are. Bits beyond the end of
// the sequence are zero, so that k-mers reaching past the end are never valid.

FISK_NOINLINE_COLD
inline void throw_invalid_validity_size_()
{
    throw std::invalid_argument(
        "Invalid call to checked k-mer extraction with validity bits that do not match the "
        "sequence length"
    );
}

template <Encoding E, Layout L>
FISK_ALWAYS_INLINE
inline void check_validity_size_(PackedSequenceWithValidity<E, L> const& seq)
{
    if (seq.valid.size() != (seq.length + 7) / 8) {
        throw_invalid_validity_size_();
    }
}

// The validity of all k-mers of a block of 32 positions whose nucleotides are all valid. A type of
// its own, so that the checks of the k-mers fold away at compile time.
struct AllValid_ {};

// The validity window at `pos`, from the 8 bytes at `pos / 8`, which have to be in `valid`. At
// least its lower 57 bits hold nucleotides.
FISK_ALWAYS_INLINE
inline std::uint64_t load_validity_window_(std::uint8_t const* valid, std::size_t pos) noexcept
{
    std::uint64_t word;
    std::memcpy(&word, valid + pos / 8, 8);
    return word >> (pos % 8);
}

// As load_validity_window_(), for positions near the end of the `size` bytes of `valid`, with zero
// for the bits beyond them.
inline std::uint64_t load_validity_window_tail_(
    std::uint8_t const* valid, std::size_t size, std::size_t pos
) noexcept {
    std::uint64_t word = 0;
    std::memcpy(&word, valid + pos / 8, std::min<std::size_t>(8, size - pos / 8));
    return word >> (pos % 8);
}

// -------------------------------------------------------------------------------------------------
//     Block Loop
// -------------------------------------------------------------------------------------------------

// The main loop of the aligned and the SIMD extraction, see packed_simd.hpp: `step(b,
// validity_at)` extracts the k-mers that start in the `StepBytes` bytes from byte `b`, for all `b`
// below `fast_bytes`, and `validity_at(pos)` is the validity of the k-mer at `pos` for them.
//
// Unchecked, all k-mers are valid. Checked, the steps run in blocks of 8 bytes, that is, 32
// positions, and each block is tested once: with one load of the validity bits at its start, which
// covers its 32 positions and the k - 1 nucleotides after them, the low `31 + k` bits of
// `block_mask`. If they are all valid, the block runs with AllValid_. Otherwise, as well as after
// the last whole block, each step loads the windows of its k-mers. Checked, `fast_bytes` must leave
// 8 bytes of `valid` to load at `pos / 8` for each of its positions, see checked_fast_bytes_().
//
// Each copy of the step is inlined. Checked, there is one for the blocks of all valid k-mers, and
// one for all other steps. With `SeparateDirtyBlocks`, the blocks with invalid nucleotides get a
// copy of their own, whose loop has a fixed count, apart from the steps after the last whole block.
// That makes the SIMD loops faster on sequences with scattered invalid nucleotides, but not the
// aligned one, which would only grow, in each of its 32 instantiations for k.
template <bool Checked, bool SeparateDirtyBlocks, std::size_t StepBytes, typename Step>
FISK_ALWAYS_INLINE
inline void run_packed_blocks_(
    std::size_t fast_bytes, std::uint8_t const* valid, std::uint64_t block_mask, Step&& step
) {
    static_assert(8 % StepBytes == 0, "Steps have to fill a block.");
    auto const all_valid = [](std::size_t) FISK_ALWAYS_INLINE_LAMBDA { return AllValid_{}; };
    auto const window_at = [valid](std::size_t pos) FISK_ALWAYS_INLINE_LAMBDA {
        return load_validity_window_(valid, pos);
    };

    std::size_t b = 0;
    if constexpr (!Checked) {
        (void) block_mask;
        (void) window_at;
        for (; b < fast_bytes; b += StepBytes) {
            step(b, all_valid);
        }
    } else {
        auto const block_is_valid_ = [&]() FISK_ALWAYS_INLINE_LAMBDA {
            std::uint64_t block_word;
            std::memcpy(&block_word, valid + b / 2, 8);
            return (block_word & block_mask) == block_mask;
        };
        auto const block_ = [&](auto const& validity_at) FISK_ALWAYS_INLINE_LAMBDA {
            for (std::size_t i = 0; i < 8; i += StepBytes) {
                step(b + i, validity_at);
            }
        };
        if constexpr (SeparateDirtyBlocks) {
            for (; b + 8 <= fast_bytes; b += 8) {
                if (block_is_valid_()) {
                    block_(all_valid);
                } else {
                    block_(window_at);
                }
            }
            for (; b < fast_bytes; b += StepBytes) {
                step(b, window_at);
            }
        } else {
            while (b < fast_bytes) {
                if (b + 8 <= fast_bytes && block_is_valid_()) {
                    block_(all_valid);
                    b += 8;
                } else {
                    // A block with invalid nucleotides, or the steps after the last whole block.
                    std::size_t const end = std::min(b + 8, fast_bytes);
                    for (; b < end; b += StepBytes) {
                        step(b, window_at);
                    }
                }
            }
        }
    }
}

// The end of the fast region of a checked loop, see run_packed_blocks_(), for `valid_size` bytes of
// validity bits. The last window of the region is loaded at byte `fast_bytes / 2 - 1` of `valid`,
// and its 8 bytes have to be there, which ends the region 56 positions before the end of `valid`,
// earlier than the codes. The region is rounded down to an even number of bytes, so that it ends
// at a byte of `valid`.
inline std::size_t checked_fast_bytes_(std::size_t fast_bytes, std::size_t valid_size) noexcept
{
    return std::min(
        fast_bytes - fast_bytes % 2, valid_size >= 7 ? 2 * (valid_size - 7) : std::size_t{0}
    );
}

// =================================================================================================
//     Aligned
// =================================================================================================

// Emit the k-mer `kmer` at `pos`, if it is valid by its `validity`, see run_packed_blocks_():
// either AllValid_, or its window, with `kbits` the mask of its low k bits.
template <Encoding E, Layout L, typename Func, typename Validity>
FISK_ALWAYS_INLINE
inline void invoke_kmer_callback_if_valid_(
    Func& func, std::size_t pos, std::uint64_t kmer, Validity validity, std::uint64_t kbits
) {
    if constexpr (std::is_same_v<Validity, AllValid_>) {
        (void) validity;
        (void) kbits;
        invoke_kmer_callback<E, L>(func, pos, kmer);
    } else if ((validity & kbits) == kbits) {
        invoke_kmer_callback<E, L>(func, pos, kmer);
    }
}

// The k-mers after the fast region of an aligned loop, one at a time.
template <bool Checked, Encoding E, Layout L, unsigned K, typename Func>
FISK_ALWAYS_INLINE
inline void emit_kmers_packed_aligned_tail_(
    PackedSequence<E, L> const& seq, std::uint8_t const* valid, std::size_t start_pos,
    std::size_t p_max, Func& func
) {
    PackedTail_ tail_vals;
    std::size_t const tail_n = collect_kmers_packed_tail_<E, L>(
        seq, start_pos, p_max, K, tail_vals
    );
    for (std::size_t i = 0; i < tail_n; ++i) {
        assert_fits_width_(tail_vals[i], K);
        if constexpr (Checked) {
            std::uint64_t const window = load_validity_window_tail_(
                valid, (seq.length + 7) / 8, start_pos + i
            );
            invoke_kmer_callback_if_valid_<E, L>(
                func, start_pos + i, tail_vals[i], window, (std::uint64_t{1} << K) - 1
            );
        } else {
            (void) valid;
            invoke_kmer_callback<E, L>(func, start_pos + i, tail_vals[i]);
        }
    }
}

// k <= 29 fits in a single 8-byte load, independently of which of the four nucleotides within
// a byte we are starting with. This "narrow-width" case can thus be optimized compared to the
// "full-width" case below (k <= 32), which needs to read an extra byte.
//
// K is a template parameter, so that all shifts and masks are compile-time constants regardless of
// the call site; with a runtime k, this was measured up to ~1.25x slower on Clang (about the same
// on GCC). The cost is one instantiation per K in for_each_kmer_packed_aligned()'s dispatch.
//
// With `Checked`, `valid` holds the validity bits of `seq`, see run_packed_blocks_(), and only the
// valid k-mers are emitted. Otherwise, it is unused and may be null.
template <bool Checked, Encoding E, Layout L, unsigned K, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_aligned_narrow_width_(
    PackedSequence<E, L> const& seq, std::uint8_t const* valid, Func&& func
) {
    static_assert(K >= 1 && K <= 29, "K must be in [1, 29]");

    std::uint64_t const mask = (std::uint64_t{1} << (2 * K)) - 1u;
    std::uint64_t const kbits = (std::uint64_t{1} << K) - 1u;
    std::size_t const p_max = seq.length - K;
    std::size_t const full_bytes = (p_max + 1) / 4;
    std::size_t const num_bytes = (seq.length + 3) / 4;
    std::size_t fast_bytes = std::min(
        full_bytes, num_bytes >= 8 ? num_bytes - 8 : std::size_t{0}
    );
    if constexpr (Checked) {
        fast_bytes = checked_fast_bytes_(fast_bytes, (seq.length + 7) / 8);
    }

    auto const step_ = [&](std::size_t b, auto const& validity_at) FISK_ALWAYS_INLINE_LAMBDA {
        std::uint64_t word;
        std::memcpy(&word, &seq.data[b], 8);
        if constexpr (L == Layout::kMSB) {
            // 58-2K is in [0,56]; adding the local shift recovers 64-2K-2*local.
            word = byte_swap_64(word) >> (58 - 2 * K);
        } else if constexpr (L == Layout::kLSB) {
            // A little-endian load already has the bases in order.
        } else {
            static_assert(
                dependent_false_v<L>,
                "Unhandled Layout in for_each_kmer_packed_aligned_narrow_width_()"
            );
        }
        std::uint64_t v0, v1, v2, v3;
        if constexpr (L == Layout::kMSB) {
            v0 = (word >> 6) & mask;
            v1 = (word >> 4) & mask;
            v2 = (word >> 2) & mask;
            v3 = (word >> 0) & mask;
        } else if constexpr (L == Layout::kLSB) {
            v0 = (word >> 0) & mask;
            v1 = (word >> 2) & mask;
            v2 = (word >> 4) & mask;
            v3 = (word >> 6) & mask;
        } else {
            static_assert(
                dependent_false_v<L>,
                "Unhandled Layout in for_each_kmer_packed_aligned_narrow_width_()"
            );
        }

        // Prevent the compiler from optimizing away the values in the benchmarks,
        // as otherwise the compiler can reduce the sum sink computation too much,
        // leaving us with no true measurement. We need the barriers here instead
        // of in the benchmark itself, so that the compiler can reorder instructions
        // within the function; otherwise, we'd nuke benchmark performance too much,
        // leaving us with an unrealistically bad measurement.
        #if defined(FISK_BENCHMARK_SINK_MODE)
        do_not_optimize(v0);
        do_not_optimize(v1);
        do_not_optimize(v2);
        do_not_optimize(v3);
        #endif

        assert_fits_width_(v0, K);
        invoke_kmer_callback_if_valid_<E, L>(func, 4 * b + 0, v0, validity_at(4 * b + 0), kbits);
        assert_fits_width_(v1, K);
        invoke_kmer_callback_if_valid_<E, L>(func, 4 * b + 1, v1, validity_at(4 * b + 1), kbits);
        assert_fits_width_(v2, K);
        invoke_kmer_callback_if_valid_<E, L>(func, 4 * b + 2, v2, validity_at(4 * b + 2), kbits);
        assert_fits_width_(v3, K);
        invoke_kmer_callback_if_valid_<E, L>(func, 4 * b + 3, v3, validity_at(4 * b + 3), kbits);
    };
    run_packed_blocks_<Checked, false, 1>(
        fast_bytes, valid, (std::uint64_t{1} << (31 + K)) - 1u, step_
    );

    emit_kmers_packed_aligned_tail_<Checked, E, L, K>(seq, valid, 4 * fast_bytes, p_max, func);
}

// k in [30, 32] needs a 9th margin byte beyond the main 8-byte word, which breaks the narrow-width
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
    } else if constexpr (L == Layout::kLSB) {
        start = 2 * local;
    } else {
        static_assert(dependent_false_v<L>, "Unhandled Layout in aligned_boundary_window_()");
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

// As for_each_kmer_packed_aligned_narrow_width_(), for k in [30, 32].
template <bool Checked, Encoding E, Layout L, unsigned K, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_aligned_full_width_(
    PackedSequence<E, L> const& seq, std::uint8_t const* valid, Func&& func
) {
    static_assert(K >= 30 && K <= 32, "K must be in [30, 32]");

    std::uint64_t mask;
    if constexpr (K == 32) {
        mask = ~std::uint64_t{0};
    } else {
        mask = (std::uint64_t{1} << (2 * K)) - 1u;
    }
    std::uint64_t const kbits = (std::uint64_t{1} << K) - 1u;

    std::size_t const p_max = seq.length - K;
    std::size_t const full_bytes = (p_max + 1) / 4;
    std::size_t const num_bytes  = (seq.length + 3) / 4;

    std::size_t fast_bytes = (num_bytes >= 9) ? std::min(full_bytes, num_bytes - 9)
                                               : std::size_t{0};
    if constexpr (Checked) {
        fast_bytes = checked_fast_bytes_(fast_bytes, (seq.length + 7) / 8);
    }

    auto const step_ = [&](std::size_t b, auto const& validity_at) FISK_ALWAYS_INLINE_LAMBDA {
        std::uint64_t lo;
        std::memcpy(&lo, &seq.data[b], 8);
        std::uint64_t hi = static_cast<std::uint64_t>(seq.data[b + 8]);
        if constexpr (L == Layout::kMSB) {
            std::uint64_t const swapped_lo = byte_swap_64(lo);
            lo = hi << 56;
            hi = swapped_lo;
        } else if constexpr (L == Layout::kLSB) {
            // A little-endian load already has the bases in order.
        } else {
            static_assert(
                dependent_false_v<L>,
                "Unhandled Layout in for_each_kmer_packed_aligned_full_width_()"
            );
        }

        auto const v0 = aligned_boundary_window_<L, K>(hi, lo, 0, mask);
        auto const v1 = aligned_boundary_window_<L, K>(hi, lo, 1, mask);
        auto const v2 = aligned_boundary_window_<L, K>(hi, lo, 2, mask);
        auto const v3 = aligned_boundary_window_<L, K>(hi, lo, 3, mask);

        #if defined(FISK_BENCHMARK_SINK_MODE)
        do_not_optimize(v0);
        do_not_optimize(v1);
        do_not_optimize(v2);
        do_not_optimize(v3);
        #endif

        assert_fits_width_(v0, K);
        invoke_kmer_callback_if_valid_<E, L>(func, 4 * b + 0, v0, validity_at(4 * b + 0), kbits);
        assert_fits_width_(v1, K);
        invoke_kmer_callback_if_valid_<E, L>(func, 4 * b + 1, v1, validity_at(4 * b + 1), kbits);
        assert_fits_width_(v2, K);
        invoke_kmer_callback_if_valid_<E, L>(func, 4 * b + 2, v2, validity_at(4 * b + 2), kbits);
        assert_fits_width_(v3, K);
        invoke_kmer_callback_if_valid_<E, L>(func, 4 * b + 3, v3, validity_at(4 * b + 3), kbits);
    };
    run_packed_blocks_<Checked, false, 1>(
        fast_bytes, valid, (std::uint64_t{1} << (31 + K)) - 1u, step_
    );

    emit_kmers_packed_aligned_tail_<Checked, E, L, K>(seq, valid, 4 * fast_bytes, p_max, func);
}

// The dispatch of for_each_kmer_packed_aligned() and for_each_kmer_packed_checked_aligned() to the
// loops of each k, see for_each_kmer_packed_aligned_narrow_width_() for `Checked` and `valid`.
template <bool Checked, Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_aligned_(
    PackedSequence<E, L> const& seq, std::uint8_t const* valid, std::size_t k, Func&& func
) {
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.length < k) {
        return;
    }

    // Local X-macro, undefined right below: the 29 narrow-width cases are otherwise identical aside
    // from their K, and spelling each out longhand would bury that in repetition.
    #define FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(K_) \
        case K_: \
            for_each_kmer_packed_aligned_narrow_width_<Checked, E, L, K_>(seq, valid, func); \
            break;

    switch (k) {
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(1)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(2)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(3)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(4)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(5)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(6)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(7)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(8)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(9)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(10)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(11)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(12)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(13)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(14)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(15)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(16)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(17)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(18)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(19)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(20)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(21)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(22)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(23)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(24)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(25)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(26)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(27)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(28)
        FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(29)
        case 30: {
            for_each_kmer_packed_aligned_full_width_<Checked, E, L, 30>(seq, valid, func);
            break;
        }
        case 31: {
            for_each_kmer_packed_aligned_full_width_<Checked, E, L, 31>(seq, valid, func);
            break;
        }
        case 32: {
            for_each_kmer_packed_aligned_full_width_<Checked, E, L, 32>(seq, valid, func);
            break;
        }
    }

    #undef FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_
}

/**
 * @brief Extract all k-mers for k in [1, 32] directly from a PackedSequence, and call a
 * callback on each. The recommended, fastest variant in this file.
 *
 * Internally dispatches to a separate implementation for each k, so that the extraction loop is
 * fully specialized for whichever k is used at runtime.
 *
 * Func may be called either as `func(pos, kmer)` or as `func(kmer)`, where `pos` is the start
 * position of the k-mer in the sequence. See invoke_kmer_callback().
 */
template <Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_aligned(
    PackedSequence<E, L> const& seq, std::size_t k, Func&& func
) {
    for_each_kmer_packed_aligned_<false>(seq, nullptr, k, func);
}

/**
 * @brief Extract the valid k-mers for k in [1, 32] directly from a PackedSequenceWithValidity, and
 * call a callback on each.
 *
 * As for_each_kmer_packed_aligned(), emitting only the k-mers whose nucleotides are all valid by
 * the validity bits of `seq`, see pack_sequence_with_validity(). The loop tests a block of 32
 * positions at once, so that sequences whose invalid nucleotides come in runs, such as the N runs
 * of assemblies, run at about the speed of the unchecked loop.
 *
 * @throws std::invalid_argument if `seq.valid` does not have `(seq.length + 7) / 8` bytes.
 */
template <Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_checked_aligned(
    PackedSequenceWithValidity<E, L> const& seq, std::size_t k, Func&& func
) {
    check_validity_size_(seq);
    for_each_kmer_packed_aligned_<true>(seq, seq.valid.data(), k, func);
}

// =================================================================================================
//     Rolling
// =================================================================================================

// The validity bits of the checked rolling loops, folded in one byte per 8 nucleotides next to the
// codes: once the byte of nucleotide `e` is folded, its bit is at 56 + e % 8, and the k-mer ending
// at `e` has its k bits right below. The accumulator starts with all bits invalid, so that the
// k-mers ending at `e < k - 1` are invalid without a test of their own.
struct RollingValidity_
{
    std::uint8_t const* valid;
    std::size_t k;
    std::uint64_t kbits;
    std::uint64_t acc = 0;

    // Before the nucleotides of byte `b` of the codes.
    FISK_ALWAYS_INLINE
    void fold(std::size_t b) noexcept
    {
        if (b % 2 == 0) {
            acc = (acc >> 8) | (std::uint64_t{valid[b / 2]} << 56);
        }
    }

    FISK_ALWAYS_INLINE
    bool kmer_ending_at_is_valid(std::size_t e) const noexcept
    {
        return ((acc >> (57 + e % 8 - k)) & kbits) == kbits;
    }
};

// k <= 29 fits in a single 64-bit accumulator, folded one byte at a time. Each byte's fold
// depends serially on the previous one, as opposed to the above independent loads.
// That serial dependency chain is what makes this version slower.
//
// `Checked` and `valid` are as for for_each_kmer_packed_aligned_narrow_width_(), see
// RollingValidity_.
template <bool Checked, Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_rolling_narrow_width_(
    PackedSequence<E, L> const& seq, std::uint8_t const* valid, std::size_t k, Func&& func
) {
    std::uint64_t const mask = (std::uint64_t{1} << (2 * k)) - 1u;
    RollingValidity_ validity{valid, k, (std::uint64_t{1} << k) - 1u};

    std::uint64_t acc = 0;

    auto fold_byte_ = [&](std::uint8_t byte) {
        if constexpr (L == Layout::kMSB) {
            acc = (acc << 8) | byte;
        } else if constexpr (L == Layout::kLSB) {
            acc = (acc >> 8) | (std::uint64_t{byte} << 56);
        } else {
            static_assert(
                dependent_false_v<L>,
                "Unhandled Layout in for_each_kmer_packed_rolling_narrow_width_()"
            );
        }
    };

    std::size_t const num_bytes = (seq.length + 3) / 4;
    for (std::size_t b = 0; b < num_bytes; ++b) {
        fold_byte_(seq.data[b]);
        if constexpr (Checked) {
            validity.fold(b);
        }

        // The 4 k-mers ending within this byte, oldest (local 0) to newest (local 3).
        for (unsigned local = 0; local < 4; ++local) {
            std::size_t const e = 4 * b + local; // base end position of this k-mer
            if (e >= seq.length) {
                break;
            }
            if constexpr (Checked) {
                if (!validity.kmer_ending_at_is_valid(e)) {
                    continue;
                }
            } else if (e + 1 < k) {
                continue;
            }

            unsigned const s = 6 - 2 * local;
            unsigned shift;
            if constexpr (L == Layout::kMSB) {
                shift = s;
            } else if constexpr (L == Layout::kLSB) {
                shift = 64 - 2 * static_cast<unsigned>(k) - s;
            } else {
                static_assert(
                    dependent_false_v<L>,
                    "Unhandled Layout in for_each_kmer_packed_rolling_narrow_width_()"
                );
            }
            auto const word = (acc >> shift) & mask;
            assert_fits_width_(word, k);
            invoke_kmer_callback<E, L>(func, e + 1 - k, word);
        }
    }
}

// k > 29 needs more than 64 bits of accumulator, so this variant folds into a hi:lo pair instead
// of the narrow-width impl's single 64-bit `acc`; correct for any k in [1, 32], just with more
// bookkeeping per byte, which is why the narrow-width impl above still exists as the faster choice
// for k <= 29.
template <bool Checked, Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_rolling_full_width_(
    PackedSequence<E, L> const& seq, std::uint8_t const* valid, std::size_t k, Func&& func
) {
    std::uint64_t const mask = (k == 32)
        ? ~std::uint64_t{0}
        : ((std::uint64_t{1} << (2 * k)) - 1u);
    RollingValidity_ validity{valid, k, (std::uint64_t{1} << k) - 1u};

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
        } else if constexpr (L == Layout::kLSB) {
            start = 128 - 2 * static_cast<unsigned>(k) - s;
        } else {
            static_assert(
                dependent_false_v<L>,
                "Unhandled Layout in for_each_kmer_packed_rolling_full_width_()"
            );
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
        } else if constexpr (L == Layout::kLSB) {
            std::uint64_t const carry = hi & 0xFFu;
            hi = (hi >> 8) | (std::uint64_t{byte} << 56);
            lo = (lo >> 8) | (carry << 56);
        } else {
            static_assert(
                dependent_false_v<L>,
                "Unhandled Layout in for_each_kmer_packed_rolling_full_width_()"
            );
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
        if constexpr (Checked) {
            validity.fold(b);
        }

        for (unsigned local = 0; local < 4; ++local) {
            std::size_t const e = 4 * b + local;
            if (e >= seq.length) {
                break;
            }
            if constexpr (Checked) {
                if (!validity.kmer_ending_at_is_valid(e)) {
                    continue;
                }
            } else if (e + 1 < k) {
                continue;
            }
            auto const word = window_(plans[local]);
            assert_fits_width_(word, k);
            invoke_kmer_callback<E, L>(func, e + 1 - k, word);
        }
    }
}

// The dispatch of for_each_kmer_packed_rolling() and for_each_kmer_packed_checked_rolling(), see
// for_each_kmer_packed_aligned_narrow_width_() for `Checked` and `valid`.
template <bool Checked, Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_rolling_(
    PackedSequence<E, L> const& seq, std::uint8_t const* valid, std::size_t k, Func&& func
) {
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.length < k) {
        return;
    }
    if (k <= 29) {
        for_each_kmer_packed_rolling_narrow_width_<Checked>(seq, valid, k, func);
    } else {
        for_each_kmer_packed_rolling_full_width_<Checked>(seq, valid, k, func);
    }
}

/**
 * @brief Extract all k-mers for k in [1, 32] from a PackedSequence via a rolling
 * accumulator, folding one byte at a time.
 *
 * Kept only as a slower algorithmic comparison baseline; prefer for_each_kmer_packed_aligned()
 * for production use.
 *
 * For k <= 29, delegates to for_each_kmer_packed_rolling_narrow_width_() (single 64-bit
 * accumulator); for k in [30, 32], to for_each_kmer_packed_rolling_full_width_() (hi:lo pair)
 * instead, since a single 64-bit accumulator no longer fits.
 *
 * Func may be called either as `func(pos, kmer)` or as `func(kmer)`, where `pos` is the start
 * position of the k-mer in the sequence. See invoke_kmer_callback().
 */
template <Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_rolling(
    PackedSequence<E, L> const& seq, std::size_t k, Func&& func
) {
    for_each_kmer_packed_rolling_<false>(seq, nullptr, k, func);
}

/**
 * @brief Extract the valid k-mers for k in [1, 32] directly from a PackedSequenceWithValidity,
 * using a rolling accumulator.
 *
 * As for_each_kmer_packed_rolling(), emitting only the k-mers whose nucleotides are all valid, as
 * for_each_kmer_packed_checked_aligned(). The validity bits roll along with the codes. Kept as a
 * baseline as well; prefer for_each_kmer_packed_checked_aligned().
 *
 * @throws std::invalid_argument if `seq.valid` does not have `(seq.length + 7) / 8` bytes.
 */
template <Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_checked_rolling(
    PackedSequenceWithValidity<E, L> const& seq, std::size_t k, Func&& func
) {
    check_validity_size_(seq);
    for_each_kmer_packed_rolling_<true>(seq, seq.valid.data(), k, func);
}

} // namespace fisk
