#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <utility>

#include "fisk/core/cast.hpp"
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

// -------------------------------------------------------------------------------------------------
//     Validity
// -------------------------------------------------------------------------------------------------

// The checked functions read the validity bits of a PackedSequenceWithValidity, see core/types.hpp,
// as windows of 64 nucleotides: bit `j` of the window at `pos` is set if nucleotide `pos + j` is
// valid, and a k-mer at `pos` is valid if bits 0 to k - 1 of its window are. Bits beyond the end of
// the sequence are zero, so that k-mers reaching past the end are never valid. The block loop below
// tests the k-mers of a window all at once, as its valid starts: bit `j` is set if the k-mer at
// `pos + j` is valid.

// The sequence of `seq` without its validity bits, for the extraction of all of its k-mers, which
// ignores them.
template <PackedSequenceType Seq>
FISK_ALWAYS_INLINE
inline auto const& without_validity_(Seq const& seq) noexcept
{
    if constexpr (PackedSequenceWithValidityType<Seq>) {
        return seq.sequence();
    } else {
        return seq;
    }
}

// The validity of k-mers whose nucleotides are all valid, for the steps of the loops below in
// which they all are. A type of its own, so that the checks of the k-mers fold away at compile
// time.
struct AllValid {};

// The form of the validity that the block loop below hands to its steps for k-mers that are not
// all valid: the valid starts of contiguous k-mers, or the validity window of the nucleotides, for
// spaced k-mers, which are tested per mask against the positions that it keeps.
enum class PackedValidity
{
    kStarts,
    kNucleotides
};

// The 8 bytes at `p` as a word with the bases in order: the first base is in the lowest bits for
// kLSB, and in the highest bits for kMSB.
template <Layout L>
FISK_ALWAYS_INLINE
inline std::uint64_t load_packed_word_(std::uint8_t const* p) noexcept
{
    std::uint64_t word;
    std::memcpy(&word, p, 8);
    if constexpr (L == Layout::kMSB) {
        return byte_swap_64(word);
    } else if constexpr (L == Layout::kLSB) {
        return word;
    } else {
        static_assert(dependent_false_v<L>, "Unhandled Layout in load_packed_word_()");
    }
}

// -------------------------------------------------------------------------------------------------
//     Block Loop
// -------------------------------------------------------------------------------------------------

// The main loop of the aligned and the SIMD extraction, see packed_simd.hpp, and of the spaced
// k-mer extraction, see kmer_spaced/packed.hpp: `step(b, validity_at)` extracts the k-mers that
// start in the `StepBytes` bytes from byte `b`, and `validity_at(pos)` is the validity of the k-mers
// from `pos` on for them: either AllValid, or a word whose bit 0 is set if the k-mer at `pos` is
// valid, see packed_kmer_is_valid_(). With PackedValidity::kNucleotides, the word is the validity
// window at `pos` instead, see Validity above, which holds at least the k nucleotides from `pos`.
// The steps cover all bytes in which a k-mer of length `k` starts, and `seq` has to have at least
// `k` nucleotides. The words that the steps load reach past the last k-mer, into the padding after
// the codes and after the validity bits, see PackedSequence (core/types.hpp).
//
// Without validity bits, all k-mers are valid. The steps whose k-mers are all within the sequence
// run with AllValid, and the step after them, if any, with a word that is all ones up to the last
// k-mer, at `p_max`, and zero after it.
//
// With validity bits, see PackedSequenceWithValidityType (core/types.hpp), the steps run in blocks
// of 8 bytes, that is, 32 positions, and each block is tested once: with one load of the validity
// bits at its start, which covers its 32 positions and the k - 1 nucleotides after them, the low
// `31 + k` bits of `block_mask`. If they are all valid, the block runs with AllValid. Otherwise, the
// block computes the valid starts of that window once, see Validity above, which hold the validity
// of all of its k-mers: the word at `pos` has bit `j` set if the k-mer at `pos + j` is valid, for
// the k-mers up to the end of the block, so that a vector of consecutive k-mers takes its valid
// lanes from the low bits. After the last whole block, each step computes the valid starts of the
// windows of its k-mers instead. The validity bits are zero after the end of the sequence, which
// makes its last k-mers invalid without a test of their own.
//
// Each copy of the step is inlined. Without validity bits, there are two: for the k-mers within
// the sequence, and for the step after them. With validity bits, there are three: for the blocks
// of all valid k-mers, for the other blocks, whose loop has a fixed count, and for the steps after
// the last whole block.
template <
    std::size_t StepBytes, PackedValidity Form = PackedValidity::kStarts,
    PackedSequenceType Seq, typename Step
>
FISK_ALWAYS_INLINE
inline void run_packed_blocks_(Seq const& seq, std::size_t k, Step&& step)
{
    static_assert(8 % StepBytes == 0, "Steps have to fill a block.");
    assert(k >= 1 && seq.length() >= k);

    // The last k-mer starts at `p_max`, in the byte before `end_bytes`.
    std::size_t const p_max = seq.length() - k;
    std::size_t const end_bytes = p_max / 4 + 1;

    // The validity of the k-mers of a step whose k-mers are all valid, see AllValid.
    auto const all_valid = [](std::size_t) FISK_ALWAYS_INLINE_LAMBDA { return AllValid{}; };

    std::size_t b = 0;
    if constexpr (!PackedSequenceWithValidityType<Seq>) {
        // The steps before `whole_bytes` have all their k-mers within the sequence. The step after
        // them, if any, ends at `end_bytes`.
        std::size_t const whole_bytes = (p_max + 1) / 4 / StepBytes * StepBytes;
        for (; b < whole_bytes; b += StepBytes) {
            step(b, all_valid);
        }
        if (b < end_bytes) {
            step(b, [p_max](std::size_t pos) FISK_ALWAYS_INLINE_LAMBDA {
                return pos <= p_max ? ~std::uint64_t{0} : std::uint64_t{0};
            });
        }
    } else {
        std::uint8_t const* const valid = seq.validity().data();
        std::uint64_t const block_mask = (std::uint64_t{1} << (31 + k)) - 1u;

        // The validity window at `pos`, from the 8 bytes at `pos / 8`. At least its lower 57 bits
        // hold nucleotides.
        auto const window_at = [valid](std::size_t pos) FISK_ALWAYS_INLINE_LAMBDA {
            std::uint64_t word;
            std::memcpy(&word, valid + pos / 8, 8);
            return word >> (pos % 8);
        };

        // The valid starts of a window `w`: bit `j` is set if bits `j` to `j + k - 1` of `w` are.
        // Each round of the loop doubles the length `len` of the runs of set bits that the bits
        // of `w` stand for, up to a power of two, and the last AND covers the remaining `k - len`
        // bits with a run that overlaps the first one. With PackedValidity::kNucleotides, the
        // window itself.
        auto const valid_starts_ = [k](std::uint64_t w) FISK_ALWAYS_INLINE_LAMBDA {
            if constexpr (Form == PackedValidity::kStarts) {
                std::size_t len = 1;
                for (; 2 * len <= k; len *= 2) {
                    w &= w >> len;
                }
                return w & (w >> (k - len));
            } else {
                (void) k;
                return w;
            }
        };

        // The steps of the block at `b`, with the given validity.
        auto const block_ = [&](auto const& validity_at) FISK_ALWAYS_INLINE_LAMBDA {
            for (std::size_t i = 0; i < 8; i += StepBytes) {
                step(b + i, validity_at);
            }
        };

        // The whole blocks, each with one test of the window at its start, and then the steps after
        // them.
        for (; b + 8 <= end_bytes; b += 8) {
            std::size_t const block_pos = 4 * b;
            std::uint64_t const block_window = window_at(block_pos);
            if ((block_window & block_mask) == block_mask) {
                block_(all_valid);
            } else {
                std::uint64_t const starts = valid_starts_(block_window);
                block_([starts, block_pos](std::size_t pos) FISK_ALWAYS_INLINE_LAMBDA {
                    return starts >> (pos - block_pos);
                });
            }
        }
        for (; b < end_bytes; b += StepBytes) {
            step(b, [&](std::size_t pos) FISK_ALWAYS_INLINE_LAMBDA {
                return valid_starts_(window_at(pos));
            });
        }
    }
}

// Whether the k-mer with the given `validity` is valid, see run_packed_blocks_(): either AllValid,
// or a word whose bit 0 is set for a valid k-mer.
template <typename Validity>
FISK_ALWAYS_INLINE
inline bool packed_kmer_is_valid_(Validity validity) noexcept
{
    if constexpr (std::is_same_v<Validity, AllValid>) {
        (void) validity;
        return true;
    } else {
        return validity & 1u;
    }
}

// =================================================================================================
//     Aligned
// =================================================================================================

// Emit the k-mer `kmer` at `pos`, if it is valid by its `validity`, see packed_kmer_is_valid_().
template <Encoding E, Layout L, typename Func, typename Validity>
FISK_ALWAYS_INLINE
inline void invoke_kmer_callback_if_valid_(
    Func& func, std::size_t pos, std::uint64_t kmer, Validity validity
) {
    if (packed_kmer_is_valid_(validity)) {
        invoke_kmer_callback<E, L>(func, pos, kmer);
    }
}

// k <= 29 fits in a single 8-byte load, independently of which of the four nucleotides within
// a byte we are starting with. This "narrow-width" case can thus be optimized compared to the
// "full-width" case below (k <= 32), which needs to read an extra byte.
//
// K is a template parameter, so that all shifts and masks are compile-time constants regardless of
// the call site, with one instantiation per K in for_each_kmer_packed_aligned()'s dispatch.
//
// For a sequence with validity bits, see run_packed_blocks_(), only the valid k-mers are emitted.
template <unsigned K, PackedSequenceType Seq, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_aligned_narrow_width_(Seq const& seq, Func&& func)
{
    static_assert(K >= 1 && K <= 29, "K must be in [1, 29]");
    constexpr Encoding E = Seq::encoding;
    constexpr Layout L = Seq::layout;

    // The 2k bits of a k-mer in the codes.
    std::uint64_t const mask = width_mask_(K);

    // The codes, read up into their padding, see run_packed_blocks_().
    std::uint8_t const* const codes = seq.codes().data();

    // One step: the 4 k-mers that start in byte `b`, from one 8-byte load, each shifted to the low
    // bits and masked. `validity_at` gives their validity, see run_packed_blocks_().
    auto const step_ = [&](std::size_t b, auto const& validity_at) FISK_ALWAYS_INLINE_LAMBDA {
        std::uint64_t word = load_packed_word_<L>(codes + b);
        if constexpr (L == Layout::kMSB) {
            // 58-2K is in [0,56]; adding the local shift recovers 64-2K-2*local.
            word >>= 58 - 2 * K;
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
        invoke_kmer_callback_if_valid_<E, L>(func, 4 * b + 0, v0, validity_at(4 * b + 0));
        assert_fits_width_(v1, K);
        invoke_kmer_callback_if_valid_<E, L>(func, 4 * b + 1, v1, validity_at(4 * b + 1));
        assert_fits_width_(v2, K);
        invoke_kmer_callback_if_valid_<E, L>(func, 4 * b + 2, v2, validity_at(4 * b + 2));
        assert_fits_width_(v3, K);
        invoke_kmer_callback_if_valid_<E, L>(func, 4 * b + 3, v3, validity_at(4 * b + 3));
    };

    // The steps over all bytes in which a k-mer starts, see run_packed_blocks_().
    run_packed_blocks_<1>(seq, K, step_);
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
template <unsigned K, PackedSequenceType Seq, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_aligned_full_width_(Seq const& seq, Func&& func)
{
    static_assert(K >= 30 && K <= 32, "K must be in [30, 32]");
    constexpr Encoding E = Seq::encoding;
    constexpr Layout L = Seq::layout;

    // As in the narrow-width loop.
    std::uint64_t const mask = width_mask_(K);
    std::uint8_t const* const codes = seq.codes().data();

    // One step: the 4 k-mers that start in byte `b`, from an 8-byte load and the byte after it,
    // combined by aligned_boundary_window_(). `validity_at` as in the narrow-width loop.
    auto const step_ = [&](std::size_t b, auto const& validity_at) FISK_ALWAYS_INLINE_LAMBDA {
        std::uint64_t const word = load_packed_word_<L>(codes + b);
        std::uint64_t const ninth = codes[b + 8];
        std::uint64_t hi;
        std::uint64_t lo;
        if constexpr (L == Layout::kMSB) {
            hi = word;
            lo = ninth << 56;
        } else if constexpr (L == Layout::kLSB) {
            hi = ninth;
            lo = word;
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
        invoke_kmer_callback_if_valid_<E, L>(func, 4 * b + 0, v0, validity_at(4 * b + 0));
        assert_fits_width_(v1, K);
        invoke_kmer_callback_if_valid_<E, L>(func, 4 * b + 1, v1, validity_at(4 * b + 1));
        assert_fits_width_(v2, K);
        invoke_kmer_callback_if_valid_<E, L>(func, 4 * b + 2, v2, validity_at(4 * b + 2));
        assert_fits_width_(v3, K);
        invoke_kmer_callback_if_valid_<E, L>(func, 4 * b + 3, v3, validity_at(4 * b + 3));
    };

    // As in the narrow-width loop.
    run_packed_blocks_<1>(seq, K, step_);
}

// The dispatch of for_each_kmer_packed_aligned() and for_each_kmer_packed_checked_aligned() to the
// loops of each k.
template <PackedSequenceType Seq, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_aligned_(Seq const& seq, std::size_t k, Func&& func)
{
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.length() < k) {
        return;
    }

    // Local X-macro, undefined right below: the 29 narrow-width cases are otherwise identical aside
    // from their K, and spelling each out longhand would bury that in repetition.
    #define FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_(K_) \
        case K_: \
            for_each_kmer_packed_aligned_narrow_width_<K_>(seq, func); \
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
            for_each_kmer_packed_aligned_full_width_<30>(seq, func);
            break;
        }
        case 31: {
            for_each_kmer_packed_aligned_full_width_<31>(seq, func);
            break;
        }
        case 32: {
            for_each_kmer_packed_aligned_full_width_<32>(seq, func);
            break;
        }
    }

    #undef FISK_PACKED_ALIGNED_NARROW_WIDTH_CASE_
}

/**
 * @brief Extract all k-mers for k in [1, 32] directly from a PackedSequence, and call a
 * callback on each. The recommended, fastest variant in this file.
 *
 * A PackedSequenceWithValidity is taken as well, ignoring its validity bits, see
 * for_each_kmer_packed_checked_aligned() for its valid k-mers only.
 *
 * Internally dispatches to a separate implementation for each k, so that the extraction loop is
 * fully specialized for whichever k is used at runtime.
 *
 * Func may be called either as `func(pos, kmer)` or as `func(kmer)`, where `pos` is the start
 * position of the k-mer in the sequence. See invoke_kmer_callback().
 */
template <PackedSequenceType Seq, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_aligned(Seq const& seq, std::size_t k, Func&& func)
{
    for_each_kmer_packed_aligned_(without_validity_(seq), k, func);
}

/**
 * @brief Extract the valid k-mers for k in [1, 32] directly from a PackedSequenceWithValidity, and
 * call a callback on each.
 *
 * As for_each_kmer_packed_aligned(), emitting only the k-mers whose nucleotides are all valid by
 * the validity bits of `seq`, see pack_sequence_with_validity(). The loop tests a block of 32
 * positions at once, and the blocks whose nucleotides are all valid run without a test per k-mer,
 * as do most blocks of sequences whose invalid nucleotides come in runs, such as assemblies.
 */
template <Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_checked_aligned(
    PackedSequenceWithValidity<E, L> const& seq, std::size_t k, Func&& func
) {
    for_each_kmer_packed_aligned_(seq, k, func);
}

// =================================================================================================
//     Rolling
// =================================================================================================

// The rolling loops fold one byte of codes at a time into an accumulator, and extract the four
// k-mers that end in it. Each fold depends serially on the previous one, as opposed to the
// independent loads of the aligned loops above, which is what makes them slower.
//
// For a sequence with validity bits, see PackedSequenceWithValidityType, only the valid k-mers are
// emitted. The validity bits are folded into an accumulator of their own, one byte per 8
// nucleotides next to the codes: once the byte of nucleotide `e` is folded, its bit is at
// 56 + e % 8, and the k-mer ending at `e` has its k bits right below. The accumulator starts with
// all bits invalid, so that the k-mers ending before position k - 1 are invalid without a test of
// their own.

// k <= 29 fits in a single 64-bit accumulator.
template <PackedSequenceType Seq, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_rolling_narrow_width_(Seq const& seq, std::size_t k, Func&& func)
{
    constexpr Encoding E = Seq::encoding;
    constexpr Layout L = Seq::layout;
    constexpr bool Checked = PackedSequenceWithValidityType<Seq>;

    // The 2k bits of a k-mer in the codes, and its k bits in the validity bits.
    std::uint64_t const mask = width_mask_(k);
    std::uint64_t const kbits = (std::uint64_t{1} << k) - 1u;

    // The codes of the last 8 bytes folded, in the order of load_packed_word_(), and the validity
    // bits, see above.
    std::uint64_t acc = 0;
    std::uint64_t valid_acc = 0;

    // The validity bits, loaded once, as the callback could change `seq` as far as the compiler
    // knows.
    std::uint8_t const* valid = nullptr;
    if constexpr (Checked) {
        valid = seq.validity().data();
    }

    // Shift a byte of codes in at the end of the newest bases: the low bits for kMSB, and the high
    // bits for kLSB, whose bytes hold their first base in the low bits.
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

    // With validity bits, shift in those of the nucleotides of byte `b` of the codes, before
    // extracting its k-mers. Byte `b / 2` of the validity bits covers the codes of bytes `b` and
    // `b + 1` for an even `b`, and is shifted in once, at the first of them.
    auto fold_valid_ = [&]([[maybe_unused]] std::size_t b) FISK_ALWAYS_INLINE_LAMBDA {
        if constexpr (Checked) {
            if (b % 2 == 0) {
                valid_acc = (valid_acc >> 8) | (std::uint64_t{valid[b / 2]} << 56);
            }
        }
    };

    // Fold one byte of codes at a time, and emit the 4 k-mers that end in it. Checked, the k-mers
    // with invalid nucleotides are skipped, and unchecked, the ones that would start before 0.
    std::size_t const num_bytes = (seq.length() + 3) / 4;
    for (std::size_t b = 0; b < num_bytes; ++b) {
        fold_byte_(seq.codes()[b]);
        fold_valid_(b);

        // The 4 k-mers ending within this byte, oldest (local 0) to newest (local 3).
        for (unsigned local = 0; local < 4; ++local) {
            std::size_t const e = 4 * b + local; // base end position of this k-mer
            if (e >= seq.length()) {
                break;
            }
            if constexpr (Checked) {
                if (((valid_acc >> (57 + e % 8 - k)) & kbits) != kbits) {
                    continue;
                }
            } else if (e + 1 < k) {
                continue;
            }

            // Shift the k-mer to the low bits: its last base is at bit `6 - 2 * local` for kMSB,
            // and its first base at bit `58 - 2k + 2 * local` for kLSB.
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
template <PackedSequenceType Seq, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_rolling_full_width_(Seq const& seq, std::size_t k, Func&& func)
{
    constexpr Encoding E = Seq::encoding;
    constexpr Layout L = Seq::layout;
    constexpr bool Checked = PackedSequenceWithValidityType<Seq>;

    // The 2k bits of a k-mer in the codes, and its k bits in the validity bits.
    std::uint64_t const mask = width_mask_(k);
    std::uint64_t const kbits = (std::uint64_t{1} << k) - 1u;

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

    // The codes of the last 16 bytes folded, as in the narrow-width loop, split into two words,
    // and the validity bits.
    std::uint64_t hi = 0;
    std::uint64_t lo = 0;
    std::uint64_t valid_acc = 0;

    // As in the narrow-width loop.
    std::uint8_t const* valid = nullptr;
    if constexpr (Checked) {
        valid = seq.validity().data();
    }

    // As in the narrow-width loop, moving the byte that leaves one word over into the other.
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

    // The k-mer of a plan, shifted to the low bits from `hi`, `lo`, or both, and masked.
    auto window_ = [&](LocalPlan const& p) -> std::uint64_t {
        std::uint64_t bits;
        switch (p.case_id) {
            case 0:  bits = hi >> (p.start - 64); break;
            case 1:  bits = lo >> p.start; break;
            default: bits = (lo >> p.start) | (hi << (64 - p.start)); break;
        }
        return bits & mask;
    };

    // As in the narrow-width loop.
    auto fold_valid_ = [&]([[maybe_unused]] std::size_t b) FISK_ALWAYS_INLINE_LAMBDA {
        if constexpr (Checked) {
            if (b % 2 == 0) {
                valid_acc = (valid_acc >> 8) | (std::uint64_t{valid[b / 2]} << 56);
            }
        }
    };

    // Fold one byte of codes at a time, and emit the 4 k-mers that end in it. Checked, the k-mers
    // with invalid nucleotides are skipped, and unchecked, the ones that would start before 0.
    std::size_t const num_bytes = (seq.length() + 3) / 4;
    for (std::size_t b = 0; b < num_bytes; ++b) {
        fold_byte_(seq.codes()[b]);
        fold_valid_(b);

        for (unsigned local = 0; local < 4; ++local) {
            std::size_t const e = 4 * b + local;
            if (e >= seq.length()) {
                break;
            }
            if constexpr (Checked) {
                if (((valid_acc >> (57 + e % 8 - k)) & kbits) != kbits) {
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

// The dispatch of for_each_kmer_packed_rolling() and for_each_kmer_packed_checked_rolling().
template <PackedSequenceType Seq, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_rolling_(Seq const& seq, std::size_t k, Func&& func)
{
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.length() < k) {
        return;
    }
    if (k <= 29) {
        for_each_kmer_packed_rolling_narrow_width_(seq, k, func);
    } else {
        for_each_kmer_packed_rolling_full_width_(seq, k, func);
    }
}

/**
 * @brief Extract all k-mers for k in [1, 32] from a PackedSequence via a rolling
 * accumulator, folding one byte at a time.
 *
 * Kept only as a slower algorithmic comparison baseline; prefer for_each_kmer_packed_aligned()
 * for production use. As that, takes a PackedSequenceWithValidity as well, ignoring its validity.
 *
 * For k <= 29, delegates to for_each_kmer_packed_rolling_narrow_width_() (single 64-bit
 * accumulator); for k in [30, 32], to for_each_kmer_packed_rolling_full_width_() (hi:lo pair)
 * instead, since a single 64-bit accumulator no longer fits.
 *
 * Func may be called either as `func(pos, kmer)` or as `func(kmer)`, where `pos` is the start
 * position of the k-mer in the sequence. See invoke_kmer_callback().
 */
template <PackedSequenceType Seq, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_rolling(Seq const& seq, std::size_t k, Func&& func)
{
    for_each_kmer_packed_rolling_(without_validity_(seq), k, func);
}

/**
 * @brief Extract the valid k-mers for k in [1, 32] directly from a PackedSequenceWithValidity,
 * using a rolling accumulator.
 *
 * As for_each_kmer_packed_rolling(), emitting only the k-mers whose nucleotides are all valid, as
 * for_each_kmer_packed_checked_aligned(). The validity bits roll along with the codes. Kept as a
 * baseline as well; prefer for_each_kmer_packed_checked_aligned().
 */
template <Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_checked_rolling(
    PackedSequenceWithValidity<E, L> const& seq, std::size_t k, Func&& func
) {
    for_each_kmer_packed_rolling_(seq, k, func);
}

} // namespace fisk
