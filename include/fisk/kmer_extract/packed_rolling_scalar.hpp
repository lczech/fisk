#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "fisk/core/cast.hpp"
#include "fisk/core/intrinsics.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/internal.hpp"

namespace fisk {

// =================================================================================================
//     K-mer Extraction from a PackedSequence, Rolling, Scalar
// =================================================================================================

// The rolling loops fold one byte of codes at a time into an accumulator, and extract the four
// k-mers that end in it. Each fold depends serially on the previous one, as opposed to the
// independent loads of the bytewise loops (packed_bytewise_scalar.hpp), which is what makes them
// slower. See there for an overview of the extraction from a PackedSequence.
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
inline void for_each_kmer_packed_rolling_scalar_narrow_width_(
    Seq const& seq, std::size_t k, Func&& func
) {
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
                "Unhandled Layout in for_each_kmer_packed_rolling_scalar_narrow_width_()"
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
                    "Unhandled Layout in for_each_kmer_packed_rolling_scalar_narrow_width_()"
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
inline void for_each_kmer_packed_rolling_scalar_full_width_(
    Seq const& seq, std::size_t k, Func&& func
) {
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
                "Unhandled Layout in for_each_kmer_packed_rolling_scalar_full_width_()"
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
                "Unhandled Layout in for_each_kmer_packed_rolling_scalar_full_width_()"
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

// The dispatch of for_each_kmer_packed_unchecked_rolling_scalar() and
// for_each_kmer_packed_checked_rolling_scalar().
template <PackedSequenceType Seq, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_rolling_scalar_(Seq const& seq, std::size_t k, Func&& func)
{
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.length() < k) {
        return;
    }
    if (k <= 29) {
        for_each_kmer_packed_rolling_scalar_narrow_width_(seq, k, func);
    } else {
        for_each_kmer_packed_rolling_scalar_full_width_(seq, k, func);
    }
}

/**
 * @brief Extract all k-mers for k in [1, 32] from a PackedSequence via a rolling
 * accumulator, folding one byte at a time.
 *
 * Kept only as a slower algorithmic comparison baseline; prefer
 * for_each_kmer_packed_unchecked_bytewise_scalar() for production use. As that, takes a
 * PackedSequenceWithValidity as well, ignoring its validity.
 *
 * For k <= 29, delegates to for_each_kmer_packed_rolling_scalar_narrow_width_() (single 64-bit
 * accumulator); for k in [30, 32], to for_each_kmer_packed_rolling_scalar_full_width_() (hi:lo
 * pair) instead, since a single 64-bit accumulator no longer fits.
 *
 * Func may be called either as `func(pos, kmer)` or as `func(kmer)`, where `pos` is the start
 * position of the k-mer in the sequence. See invoke_kmer_callback().
 */
template <PackedSequenceType Seq, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_unchecked_rolling_scalar(
    Seq const& seq, std::size_t k, Func&& func
) {
    for_each_kmer_packed_rolling_scalar_(without_validity_(seq), k, func);
}

/**
 * @brief Extract the valid k-mers for k in [1, 32] directly from a PackedSequenceWithValidity,
 * using a rolling accumulator.
 *
 * As for_each_kmer_packed_unchecked_rolling_scalar(), emitting only the k-mers whose nucleotides
 * are all valid, as for_each_kmer_packed_checked_bytewise_scalar(). The validity bits roll along
 * with the codes. Kept as a baseline as well; prefer
 * for_each_kmer_packed_checked_bytewise_scalar().
 */
template <Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_checked_rolling_scalar(
    PackedSequenceWithValidity<E, L> const& seq, std::size_t k, Func&& func
) {
    for_each_kmer_packed_rolling_scalar_(seq, k, func);
}

} // namespace fisk
