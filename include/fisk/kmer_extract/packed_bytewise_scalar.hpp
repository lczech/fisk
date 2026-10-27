#pragma once

#include <cstddef>
#include <cstdint>

#include "fisk/core/cast.hpp"
#include "fisk/util/intrinsics.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/internal.hpp"

namespace fisk {

// =================================================================================================
//     K-mer Extraction from a PackedSequence, Bytewise, Scalar
// =================================================================================================

// Unlike the extraction from ASCII input (ascii_*.hpp), which checks character validity, the
// extraction from a PackedSequence (sequence_pack/pack_sequence.hpp), in packed_*.hpp, reads the
// already-encoded bases directly. Unchecked, it assumes that `seq` holds only valid bases,
// allowing for higher performance.
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
// Two public functions cover the full documented k in [1, 32]:
// for_each_kmer_packed_unchecked_bytewise_scalar() (the fast path, below) and
// for_each_kmer_packed_unchecked_rolling_scalar() (a simpler reference baseline, not recommended
// for production, packed_rolling_scalar.hpp). The SIMD extraction is in packed_bytewise_simd.hpp.
// Both internally specialize for k <= 29 (where a single 64-bit accumulator suffices) vs k > 29
// (needing an extra byte for spilling).
//
// Their checked counterparts, for_each_kmer_packed_checked_bytewise_scalar() and
// for_each_kmer_packed_checked_rolling_scalar(), take a PackedSequenceWithValidity instead, and
// emit only the k-mers whose nucleotides are all valid by its validity bits.

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
// the call site, with one instantiation per K in for_each_kmer_packed_unchecked_bytewise_scalar()'s
// dispatch.
//
// For a sequence with validity bits, see run_packed_bytewise_blocks_(), only the valid k-mers are
// emitted.
template <unsigned K, PackedSequenceType Seq, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_bytewise_scalar_narrow_width_(Seq const& seq, Func&& func)
{
    static_assert(K >= 1 && K <= 29, "K must be in [1, 29]");
    constexpr Encoding E = Seq::encoding;
    constexpr Layout L = Seq::layout;

    // The 2k bits of a k-mer in the codes.
    std::uint64_t const mask = width_mask_(K);

    // The codes, read up into their padding, see run_packed_bytewise_blocks_().
    std::uint8_t const* const codes = seq.codes().data();

    // One step: the 4 k-mers that start in byte `b`, from one 8-byte load, each shifted to the low
    // bits and masked. `validity_at` gives their validity, see run_packed_bytewise_blocks_().
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
                "Unhandled Layout in for_each_kmer_packed_bytewise_scalar_narrow_width_()"
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

    // The steps over all bytes in which a k-mer starts, see run_packed_bytewise_blocks_().
    run_packed_bytewise_blocks_<1>(seq, K, step_);
}

// k in [30, 32] needs a 9th margin byte beyond the main 8-byte word, which breaks the narrow-width
// single-shift exploit above. These three cases instead get their own specializations, where every
// shift is a compile-time constant via the K template parameter for best optimization potential.
template <Layout L, unsigned K>
inline std::uint64_t kmer_packed_bytewise_boundary_window_(
    std::uint64_t hi, std::uint64_t lo, unsigned local, std::uint64_t mask
) {
    // Shift values are compile-time constants.
    unsigned start;
    if constexpr (L == Layout::kMSB) {
        start = 128 - 2 * local - 2 * K;
    } else if constexpr (L == Layout::kLSB) {
        start = 2 * local;
    } else {
        static_assert(
            dependent_false_v<L>, "Unhandled Layout in kmer_packed_bytewise_boundary_window_()"
        );
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

// As for_each_kmer_packed_bytewise_scalar_narrow_width_(), for k in [30, 32].
template <unsigned K, PackedSequenceType Seq, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_bytewise_scalar_full_width_(Seq const& seq, Func&& func)
{
    static_assert(K >= 30 && K <= 32, "K must be in [30, 32]");
    constexpr Encoding E = Seq::encoding;
    constexpr Layout L = Seq::layout;

    // As in the narrow-width loop.
    std::uint64_t const mask = width_mask_(K);
    std::uint8_t const* const codes = seq.codes().data();

    // One step: the 4 k-mers that start in byte `b`, from an 8-byte load and the byte after it,
    // combined by kmer_packed_bytewise_boundary_window_(). `validity_at` as in the narrow-width
    // loop.
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
                "Unhandled Layout in for_each_kmer_packed_bytewise_scalar_full_width_()"
            );
        }

        auto const v0 = kmer_packed_bytewise_boundary_window_<L, K>(hi, lo, 0, mask);
        auto const v1 = kmer_packed_bytewise_boundary_window_<L, K>(hi, lo, 1, mask);
        auto const v2 = kmer_packed_bytewise_boundary_window_<L, K>(hi, lo, 2, mask);
        auto const v3 = kmer_packed_bytewise_boundary_window_<L, K>(hi, lo, 3, mask);

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
    run_packed_bytewise_blocks_<1>(seq, K, step_);
}

// The dispatch of for_each_kmer_packed_unchecked_bytewise_scalar() and
// for_each_kmer_packed_checked_bytewise_scalar() to the loops of each k.
template <PackedSequenceType Seq, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_bytewise_scalar_(Seq const& seq, std::size_t k, Func&& func)
{
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.length() < k) {
        return;
    }

    // Local X-macro, undefined right below: the 29 narrow-width cases are otherwise identical aside
    // from their K, and spelling each out longhand would bury that in repetition.
    #define FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(K_) \
        case K_: \
            for_each_kmer_packed_bytewise_scalar_narrow_width_<K_>(seq, func); \
            break;

    switch (k) {
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(1)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(2)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(3)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(4)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(5)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(6)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(7)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(8)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(9)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(10)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(11)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(12)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(13)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(14)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(15)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(16)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(17)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(18)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(19)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(20)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(21)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(22)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(23)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(24)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(25)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(26)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(27)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(28)
        FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_(29)
        case 30: {
            for_each_kmer_packed_bytewise_scalar_full_width_<30>(seq, func);
            break;
        }
        case 31: {
            for_each_kmer_packed_bytewise_scalar_full_width_<31>(seq, func);
            break;
        }
        case 32: {
            for_each_kmer_packed_bytewise_scalar_full_width_<32>(seq, func);
            break;
        }
    }

    #undef FISK_KMER_PACKED_BYTEWISE_SCALAR_NARROW_WIDTH_CASE_
}

/**
 * @brief Extract all k-mers for k in [1, 32] directly from a PackedSequence, and call a
 * callback on each. The recommended, fastest variant in this file.
 *
 * A PackedSequenceWithValidity is taken as well, ignoring its validity bits, see
 * for_each_kmer_packed_checked_bytewise_scalar() for its valid k-mers only.
 *
 * Internally dispatches to a separate implementation for each k, so that the extraction loop is
 * fully specialized for whichever k is used at runtime.
 *
 * Func may be called either as `func(pos, kmer)` or as `func(kmer)`, where `pos` is the start
 * position of the k-mer in the sequence. See invoke_kmer_callback().
 */
template <PackedSequenceType Seq, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_unchecked_bytewise_scalar(
    Seq const& seq, std::size_t k, Func&& func
) {
    for_each_kmer_packed_bytewise_scalar_(without_validity_(seq), k, func);
}

/**
 * @brief Extract the valid k-mers for k in [1, 32] directly from a PackedSequenceWithValidity, and
 * call a callback on each.
 *
 * As for_each_kmer_packed_unchecked_bytewise_scalar(), emitting only the k-mers whose nucleotides
 * are all valid by the validity bits of `seq`, see pack_sequence_checked(). The loop tests a block
 * of 32 positions at once, and the blocks whose nucleotides are all valid run without a test per
 * k-mer, as do most blocks of sequences whose invalid nucleotides come in runs, such as assemblies.
 */
template <Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_checked_bytewise_scalar(
    PackedSequenceWithValidity<E, L> const& seq, std::size_t k, Func&& func
) {
    for_each_kmer_packed_bytewise_scalar_(seq, k, func);
}

} // namespace fisk
