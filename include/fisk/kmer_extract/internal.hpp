#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/types.hpp"

namespace fisk {

// =================================================================================================
//     K-mer Extraction, Internal Helpers
// =================================================================================================

/**
 * @brief Throw for an invalid k-mer extraction precondition.
 *
 * The extracted k-mer length must satisfy `1 <= k <= k_max`.
 * This helper is kept separate from the hot loop so the performance-critical extractor
 * can remain branch-free and avoid the runtime overhead of an inlined exception path.
 */
FISK_NOINLINE_COLD
inline void throw_invalid_kmer_k_(std::size_t k_max)
{
    // Keep the validity check out of the hot path: throwing here avoids a cold branch/landing pad
    // in otherwise branch-free extractors, which could prevent their inlining.
    throw std::invalid_argument(
        "Invalid call to k-mer extraction with k not in [1, " + std::to_string(k_max) + "]"
    );
}

// =================================================================================================
//     K-mer Extraction from a PackedSequence, Internal Helpers
// =================================================================================================

// Both MSB and LSB below load larger words from raw bytes via memcpy and rely on how the host
// interprets them as an integer. Both are therefore little-endian-specific; see
// core/platform.hpp for the assertion that enforces this.

// -------------------------------------------------------------------------------------------------
//     Details
// -------------------------------------------------------------------------------------------------

// Precondition-check throws are routed through the shared throw_invalid_kmer_k_()
// (see above) rather than thrown directly, trying to keep these functions small enough for
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

// The main loop of the bytewise scalar and SIMD extraction, see packed_bytewise_scalar.hpp and
// packed_bytewise_simd.hpp, and of the spaced k-mer extraction, see
// spaced_kmer_extract/packed_bytewise_scalar.hpp and packed_bytewise_simd.hpp: `step(b,
// validity_at)` extracts the k-mers that start in the `StepBytes` bytes from byte `b`, and
// `validity_at(pos)` is the validity of the k-mers from `pos` on for them: either AllValid, or a
// word whose bit 0 is set if the k-mer at `pos` is valid, see packed_kmer_is_valid_(). With
// PackedValidity::kNucleotides, the word is the validity window at `pos` instead, see Validity
// above, which holds at least the k nucleotides from `pos`. The steps cover all bytes in which a
// k-mer of length `k` starts, and `seq` has to have at least `k` nucleotides. The words that the
// steps load reach past the last k-mer, into the padding after the codes and after the validity
// bits, see PackedSequence (core/types.hpp).
//
// Without validity bits, all k-mers are valid. The steps whose k-mers are all within the sequence
// run with AllValid, and the step after them, if any, with a word that is all ones up to the last
// k-mer, at `p_max`, and zero after it.
//
// With validity bits, see PackedSequenceWithValidityType (core/types.hpp), the steps run in blocks
// of 8 bytes, that is, 32 positions, and each block is tested once: with one load of the validity
// bits at its start, which covers its 32 positions and the k - 1 nucleotides after them, the low
// `31 + k` bits of `block_mask`. If they are all valid, the block runs with AllValid. Otherwise,
// the block computes the valid starts of that window once, see Validity above, which hold the
// validity of all of its k-mers: the word at `pos` has bit `j` set if the k-mer at `pos + j` is
// valid, for the k-mers up to the end of the block, so that a vector of consecutive k-mers takes
// its valid lanes from the low bits. After the last whole block, each step computes the valid
// starts of the windows of its k-mers instead. The validity bits are zero after the end of the
// sequence, which makes its last k-mers invalid without a test of their own.
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
inline void run_packed_bytewise_blocks_(Seq const& seq, std::size_t k, Step&& step)
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

// Whether the k-mer with the given `validity` is valid, see run_packed_bytewise_blocks_(): either
// AllValid, or a word whose bit 0 is set for a valid k-mer.
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

} // namespace fisk
