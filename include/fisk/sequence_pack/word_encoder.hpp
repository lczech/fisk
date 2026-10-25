#pragma once

#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/simd.hpp"
#include "fisk/core/simd_scalar.hpp"
#include "fisk/core/types.hpp"
#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/bit_extract/simd.hpp"

namespace fisk {

// =================================================================================================
//     Word Encoders
// =================================================================================================

// Each of these encodes 8 ASCII characters, loaded as one little-endian word, into their packed
// 2-bit codes, for use with pack_sequence_unchecked(). They assume valid input: an invalid
// character is packed as an arbitrary code at its own position, without affecting any other
// position.
//
// WordEncoderPext and WordEncoderButterfly gather the code bits under a mask, via hardware PEXT
// or the software butterfly network. WordEncoderFold and WordEncoderDirect instead move the codes
// of each word together: the former with shifts and masks, the latter with instructions specific
// to the SIMD arch, which makes it the fastest of them. All are templated on a SIMD arch
// (core/simd.hpp), with which they encode `lanes` words at once; WordEncoderPext still uses one
// PEXT per word. WordEncoderDirectDouble is a variant of WordEncoderDirect that processes two
// vectors of the arch per step.
//
// For pack_sequence_checked(), they also store which characters are valid nucleotides, one
// bit each, see valid_nucleotide_bits(). That only depends on the loaded characters, so that their
// store_valid_bits() takes those, not the codes.

// -----------------------------------------------------------------------------
//     Helper Functions
// -----------------------------------------------------------------------------

/**
 * @brief Concept for anything usable as a word encoder for pack_sequence_unchecked().
 *
 * Requires the encoder to state both conventions its output is in, from which
 * pack_sequence_unchecked() derives the type of the PackedSequence it fills, and to encode one
 * 8-character word at a time.
 *
 * It also states the number of `lanes`, that is, words that it encodes at once. With more than one,
 * pack_sequence_unchecked() additionally uses `simd_vector`, `load_unaligned()`, `store_packed()`,
 * and the call operator on a whole vector, as provided by the encoders that take a SIMD arch, and
 * pack_sequence_checked() also `store_valid_bits()`.
 */
template <typename T>
concept WordEncoder = requires(T const encoder, std::uint64_t word) {
    { T::encoding } -> std::convertible_to<Encoding>;
    { T::layout }   -> std::convertible_to<Layout>;
    { T::lanes }    -> std::convertible_to<std::size_t>;
    { encoder(word) } -> std::convertible_to<std::uint64_t>;
};

/**
 * @brief Mask selecting the two code-bearing bits of every ASCII byte, for a given encoding.
 *
 * This is where the two bits end up after pack_prepare_codes_(): bits 0-1 of every byte for ACGT,
 * which it computes explicitly; bits 1-2 for ACTG, which sit there in the raw byte already.
 */
template <Encoding E>
inline constexpr std::uint64_t pack_mask_() noexcept
{
    if constexpr (E == Encoding::kACGT) {
        return 0x0303030303030303ULL;
    } else if constexpr (E == Encoding::kACTG) {
        return 0x0606060606060606ULL;
    } else {
        static_assert(dependent_false_v<E>, "Unhandled Encoding in pack_mask_()");
        return 0;
    }
}

template <Encoding E>
inline constexpr std::uint64_t pack_mask_v = pack_mask_<E>();

/**
 * @brief Apply the per-encoding pre-transform that puts the 2-bit codes under pack_mask_v.
 *
 * A no-op for ACTG, whose codes already sit under its mask; a SWAR bit trick for ACGT. ACTG needs
 * no shift down to bits 0-1 either: the extraction step (PEXT or the butterfly network) that
 * gathers the bits under pack_mask_v does that implicitly, as part of packing the gathered bits
 * together.
 */
template <Encoding E>
inline constexpr std::uint64_t pack_prepare_codes_(std::uint64_t word) noexcept
{
    if constexpr (E == Encoding::kACGT) {
        return ((word >> 1) ^ (word >> 2)) & pack_mask_v<E>;
    } else if constexpr (E == Encoding::kACTG) {
        return word;
    } else {
        static_assert(dependent_false_v<E>, "Unhandled Encoding in pack_prepare_codes_()");
        return 0;
    }
}

/**
 * @brief Prepare a chunk for extraction under a given layout.
 *
 * Byte-swaps for MSB, so that the chunk's first character ends up in the high bits; a no-op for
 * LSB, which is the order a little-endian load already gives.
 */
template <Layout L>
inline constexpr std::uint64_t pack_orient_chunk_(std::uint64_t word) noexcept
{
    if constexpr (L == Layout::kMSB) {
        return byte_swap_64(word);
    } else if constexpr (L == Layout::kLSB) {
        return word;
    } else {
        static_assert(dependent_false_v<L>, "Unhandled Layout in pack_orient_chunk_()");
        return 0;
    }
}

// -----------------------------------------------------------------------------
//     Nucleotide Validity
// -----------------------------------------------------------------------------

// For each value of the low four bits of a character, the one valid nucleotide with these bits,
// in lower case, or zero if there is none. Setting bit 5 of a character, which turns upper case
// into lower case, does not change its low four bits, so that a character is valid if it equals
// its entry here after setting bit 5. Zero never equals a character with bit 5 set.
inline constexpr std::array<std::uint8_t, 16> valid_nucleotide_table_ = [] {
    std::array<std::uint8_t, 16> table{};
    for (char const c : {'a', 'c', 'g', 't'}) {
        table[static_cast<std::uint8_t>(c) & 0x0F] = static_cast<std::uint8_t>(c);
    }
    return table;
}();

// Form of valid_nucleotide_bits() for archs with a byte shuffle: compare each character, set to
// lower case, to its entry in valid_nucleotide_table_.
template <SimdArchWithByteShuffle Simd>
FISK_ALWAYS_INLINE
inline std::uint64_t valid_nucleotide_bits_lookup_(typename Simd::simd_vector v) noexcept
{
    typename Simd::simd_vector const lower = Simd::bitwise_or(
        v, Simd::broadcast(0x2020202020202020ULL)
    );
    return Simd::compare_eq_bytes_to_bits(
        lower, Simd::lookup_low_nibbles(lower, valid_nucleotide_table_)
    );
}

// Form of valid_nucleotide_bits() for any arch: compare each character, set to lower case, to
// each of the four nucleotides.
template <SimdArch Simd>
FISK_ALWAYS_INLINE
inline std::uint64_t valid_nucleotide_bits_compare_(typename Simd::simd_vector v) noexcept
{
    typename Simd::simd_vector const lower = Simd::bitwise_or(
        v, Simd::broadcast(0x2020202020202020ULL)
    );
    auto const equal = [lower](char nucleotide) {
        auto const n = static_cast<std::uint64_t>(static_cast<std::uint8_t>(nucleotide));
        return Simd::compare_eq_bytes_to_bits(lower, Simd::broadcast(n * 0x0101010101010101ULL));
    };
    return equal('a') | equal('c') | equal('g') | equal('t');
}

// SWAR form of valid_nucleotide_bits() for one word. A byte of `x = c ^ n` is zero if c equals n.
// The sum within its low seven bits then has no carry into bit 7, which keeps the test exact, and
// it combines the four nucleotides with one AND before gathering the eight bits by multiplication.
FISK_ALWAYS_INLINE
inline std::uint64_t valid_nucleotide_bits_swar_(std::uint64_t word) noexcept
{
    constexpr std::uint64_t low7 = 0x7F7F7F7F7F7F7F7FULL;
    std::uint64_t const lower = word | 0x2020202020202020ULL;
    auto const nonzero = [lower](std::uint64_t nucleotide) {
        std::uint64_t const x = lower ^ (nucleotide * 0x0101010101010101ULL);
        return ((x & low7) + low7) | x;
    };
    std::uint64_t const invalid = nonzero('a') & nonzero('c') & nonzero('g') & nonzero('t');
    std::uint64_t const valid_high = ~(invalid | low7);
    return ((valid_high >> 7) * 0x0102040810204080ULL) >> 56;
}

/**
 * @brief Bit `j` is set if byte `j` of `v` is a valid nucleotide, that is, one of `ACGTacgt`.
 *
 * Validity is a property of the characters alone, so that this is the same for every Encoding and
 * Layout. Archs with a byte shuffle look up the one valid character for the low four bits of each
 * byte, the other vector archs compare to each of the four nucleotides, and SimdScalar uses SWAR.
 */
template <SimdArch Simd>
FISK_ALWAYS_INLINE
inline std::uint64_t valid_nucleotide_bits(typename Simd::simd_vector v) noexcept
{
    if constexpr (SimdArchWithByteShuffle<Simd>) {
        return valid_nucleotide_bits_lookup_<Simd>(v);
    } else if constexpr (Simd::lanes > 1) {
        return valid_nucleotide_bits_compare_<Simd>(v);
    } else {
        static_assert(std::same_as<Simd, SimdScalar>, "Unhandled SIMD arch with a single lane");
        return valid_nucleotide_bits_swar_(v);
    }
}

// Store the validity bits of the `Simd::lanes` words of `v`, one byte per word, see the
// `store_valid_bits()` of the word encoders.
template <SimdArch Simd>
FISK_ALWAYS_INLINE
inline void store_valid_nucleotide_bits_(typename Simd::simd_vector v, std::uint8_t* out) noexcept
{
    std::uint64_t const bits = valid_nucleotide_bits<Simd>(v);
    std::memcpy(out, &bits, Simd::lanes);
}

// =================================================================================================
//     Word Encoder Pext
// =================================================================================================

#if defined(FISK_HAS_BMI2)

/**
 * @brief Encode 8 ASCII characters into packed 2-bit codes via a single hardware PEXT call.
 *
 * PEXT has no vector form, so that the SIMD arch serves to process `lanes` words per step: the
 * steps before the PEXT of each word, and their validity for pack_sequence_checked(), see
 * store_valid_bits(), are done on all of them at once.
 */
template <Encoding E, Layout L, SimdArch Simd = SimdScalar>
struct WordEncoderPext
{
    static constexpr Encoding encoding = E;
    static constexpr Layout layout = L;
    static constexpr std::size_t lanes = Simd::lanes;

    // The words of one step twice: as words for their PEXT, and as a vector for the steps before it
    // and for their validity. Loading the words directly saves moving them out of the vector where
    // they need no steps before their PEXT.
    struct simd_vector
    {
        std::array<std::uint64_t, lanes> words;
        typename Simd::simd_vector vector;
    };

    static simd_vector load_unaligned(void const* a) noexcept
    {
        simd_vector x;
        std::memcpy(x.words.data(), a, sizeof(x.words));
        x.vector = Simd::load_unaligned(a);
        return x;
    }

    // Each code holds the 8 bases of its word in its low 16 bits, see write_two_bit_chunk_(). Two
    // or four of them make up one store. For MSB, the two bytes of each code have to be swapped:
    // putting the codes together in reverse order and swapping all bytes does that at once.
    static void store_packed(std::array<std::uint64_t, lanes> const& codes, char* out) noexcept
    {
        constexpr std::size_t group = lanes < 4 ? lanes : 4;
        static_assert(group == 2 || group == 4);
        for (std::size_t i = 0; i < lanes; i += group) {
            std::uint64_t packed = 0;
            if constexpr (L == Layout::kMSB) {
                for (std::size_t j = 0; j < group; ++j) {
                    packed |= codes[i + j] << (16 * (group - 1 - j));
                }
                if constexpr (group == 4) {
                    packed = byte_swap_64(packed);
                } else {
                    packed = byte_swap_32(static_cast<std::uint32_t>(packed));
                }
            } else if constexpr (L == Layout::kLSB) {
                for (std::size_t j = 0; j < group; ++j) {
                    packed |= codes[i + j] << (16 * j);
                }
            } else {
                static_assert(
                    dependent_false_v<L>, "Unhandled Layout in WordEncoderPext::store_packed()"
                );
            }
            std::memcpy(out + 2 * i, &packed, 2 * group);
        }
    }

    static void store_valid_bits(simd_vector const& x, std::uint8_t* out) noexcept
    {
        store_valid_nucleotide_bits_<Simd>(x.vector, out);
    }

    // The steps of pack_orient_chunk_() and pack_prepare_codes_() are done for all words at once,
    // and then replace the loaded words. The store forwards to the loads of the words.
    FISK_ALWAYS_INLINE
    std::array<std::uint64_t, lanes> operator()(simd_vector const& x) const noexcept
    requires (lanes > 1)
    {
        alignas(64) std::array<std::uint64_t, lanes> words = x.words;
        if constexpr (L == Layout::kMSB || E == Encoding::kACGT) {
            typename Simd::simd_vector v = x.vector;
            if constexpr (L == Layout::kMSB) {
                v = Simd::byte_swap_lanes(v);
            }
            if constexpr (E == Encoding::kACGT) {
                v = Simd::bitwise_and(
                    Simd::bitwise_xor(
                        Simd::template shift_right<1>(v), Simd::template shift_right<2>(v)
                    ),
                    Simd::broadcast(pack_mask_v<E>)
                );
            }
            Simd::store_aligned(v, words.data());
        }
        std::array<std::uint64_t, lanes> codes;
        for (std::size_t i = 0; i < lanes; ++i) {
            codes[i] = bit_extract_pext(words[i], BitExtractMask(pack_mask_v<E>));
        }
        return codes;
    }

    inline std::uint64_t operator()(std::uint64_t word) const noexcept
    {
        word = pack_prepare_codes_<E>(pack_orient_chunk_<L>(word));
        return bit_extract_pext(word, BitExtractMask(pack_mask_v<E>));
    }
};

#endif // FISK_HAS_BMI2

// =================================================================================================
//     Word Encoder Butterfly
// =================================================================================================

/**
 * @brief Same as WordEncoderPext, via the software butterfly network instead of PEXT.
 *
 * The fallback tier for hardware without fast PEXT (e.g. pre-Zen3 AMD, where PEXT/PDEP are
 * microcoded and slow), or without PEXT at all.
 *
 * With a SIMD arch other than SimdScalar (see core/simd.hpp), the call operator on a whole
 * `simd_vector` encodes `lanes` words at once, applying the same steps to each lane. The call
 * operator on a single word remains, for sequences shorter than a vector, and for the last bases
 * of a sequence that do not fill a whole word.
 *
 * The encoder is stateless: its butterfly table is a compile-time constant, so that the compiler
 * can keep it in registers. A table in the encoder object instead would have to be reloaded after
 * each store of packed bytes, which might alias it. For the same reason, the call operators are
 * forced inline, as otherwise the constants would be rebuilt on each call.
 */
template <Encoding E, Layout L, SimdArch Simd = SimdScalar>
struct WordEncoderButterfly
{
    static constexpr Encoding encoding = E;
    static constexpr Layout layout = L;
    static constexpr std::size_t lanes = Simd::lanes;
    using simd_vector = typename Simd::simd_vector;

    static constexpr BitExtractButterflyTable table =
        make_bit_extract_butterfly_table(pack_mask_v<E>);

    static simd_vector load_unaligned(void const* a) noexcept
    {
        return Simd::load_unaligned(a);
    }

    // Each lane holds the 8 bases of its chunk in its low 16 bits, see write_two_bit_chunk_().
    static void store_packed(simd_vector const& v, char* out) noexcept
    {
        if constexpr (L == Layout::kMSB) {
            Simd::template store_truncated_u16<true>(v, out);
        } else if constexpr (L == Layout::kLSB) {
            Simd::template store_truncated_u16<false>(v, out);
        } else {
            static_assert(
                dependent_false_v<L>, "Unhandled Layout in WordEncoderButterfly::store_packed()"
            );
        }
    }

    static void store_valid_bits(simd_vector const& v, std::uint8_t* out) noexcept
    {
        store_valid_nucleotide_bits_<Simd>(v, out);
    }

    // For SimdScalar, the vector is a single word, which is encoded by the call operator below.
    FISK_ALWAYS_INLINE
    simd_vector operator()(simd_vector x) const noexcept
    requires (lanes > 1)
    {
        if constexpr (L == Layout::kMSB) {
            x = Simd::byte_swap_lanes(x);
        } else if constexpr (L == Layout::kLSB) {
            // A little-endian load already has the bases in order.
        } else {
            static_assert(
                dependent_false_v<L>, "Unhandled Layout in WordEncoderButterfly::operator()"
            );
        }
        if constexpr (E == Encoding::kACGT) {
            simd_vector const s1 = Simd::template shift_right<1>(x);
            simd_vector const s2 = Simd::template shift_right<2>(x);
            simd_vector const acgt_mask = Simd::broadcast(pack_mask_v<Encoding::kACGT>);
            x = Simd::bitwise_and(Simd::bitwise_xor(s1, s2), acgt_mask);
        } else if constexpr (E == Encoding::kACTG) {
            // The codes already sit in bits 1-2 of each byte, see pack_prepare_codes_().
        } else {
            static_assert(
                dependent_false_v<E>, "Unhandled Encoding in WordEncoderButterfly::operator()"
            );
        }
        return BitExtractKernelButterfly<Simd>::bit_extract(x, table);
    }

    FISK_ALWAYS_INLINE
    std::uint64_t operator()(std::uint64_t word) const noexcept
    {
        return bit_extract_butterfly_table(
            pack_prepare_codes_<E>(pack_orient_chunk_<L>(word)), table
        );
    }
};

// =================================================================================================
//     Word Encoder Fold
// =================================================================================================

/**
 * @brief Encode ASCII characters into packed 2-bit codes by folding them together pairwise.
 *
 * Three rounds of shifting and masking move the codes next to each other: first the two codes of
 * each two bytes, then the resulting pairs, and then the two halves of each word. This only uses
 * the operations that every SIMD arch provides (see core/simd.hpp), without any code specific to
 * an arch. It is faster than WordEncoderButterfly, but slower than WordEncoderDirect on all archs
 * measured, and is kept as a simple reference to compare against.
 *
 * As for WordEncoderButterfly, the call operator on a whole `simd_vector` processes `lanes` words
 * at once, and the one on a single word the rest.
 */
template <Encoding E, Layout L, SimdArch Simd = SimdScalar>
struct WordEncoderFold
{
    static constexpr Encoding encoding = E;
    static constexpr Layout layout = L;
    static constexpr std::size_t lanes = Simd::lanes;
    using simd_vector = typename Simd::simd_vector;

    static simd_vector load_unaligned(void const* a) noexcept
    {
        return Simd::load_unaligned(a);
    }

    // Each lane holds the 8 bases of its chunk in its low 16 bits, see write_two_bit_chunk_().
    static void store_packed(simd_vector const& v, char* out) noexcept
    {
        if constexpr (L == Layout::kMSB) {
            Simd::template store_truncated_u16<true>(v, out);
        } else if constexpr (L == Layout::kLSB) {
            Simd::template store_truncated_u16<false>(v, out);
        } else {
            static_assert(
                dependent_false_v<L>, "Unhandled Layout in WordEncoderFold::store_packed()"
            );
        }
    }

    static void store_valid_bits(simd_vector const& v, std::uint8_t* out) noexcept
    {
        store_valid_nucleotide_bits_<Simd>(v, out);
    }

    // For SimdScalar, the vector is a single word, which is encoded by the call operator below.
    FISK_ALWAYS_INLINE
    simd_vector operator()(simd_vector x) const noexcept
    requires (lanes > 1)
    {
        return fold_<Simd>(x);
    }

    FISK_ALWAYS_INLINE
    std::uint64_t operator()(std::uint64_t word) const noexcept
    {
        return fold_<SimdScalar>(word) & 0xFFFF;
    }

private:

    // The result is in the low 16 bits of each lane; the bits above them are not cleared.
    template <SimdArch S>
    FISK_ALWAYS_INLINE
    static typename S::simd_vector fold_(typename S::simd_vector x) noexcept
    {
        if constexpr (L == Layout::kMSB) {
            x = S::byte_swap_lanes(x);
        } else if constexpr (L == Layout::kLSB) {
            // A little-endian load already has the bases in order.
        } else {
            static_assert(dependent_false_v<L>, "Unhandled Layout in WordEncoderFold");
        }
        if constexpr (E == Encoding::kACGT) {
            x = S::bitwise_xor(S::template shift_right<1>(x), S::template shift_right<2>(x));
        } else if constexpr (E == Encoding::kACTG) {
            // The codes already sit in bits 1-2 of each byte, see pack_prepare_codes_().
        } else {
            static_assert(dependent_false_v<E>, "Unhandled Encoding in WordEncoderFold");
        }

        // Bit offset of the codes within their bytes, which the folded codes keep until the end.
        constexpr int offset = std::countr_zero(pack_mask_v<E>);
        x = S::bitwise_and(x, S::broadcast(pack_mask_v<E>));
        x = S::bitwise_and(
            S::bitwise_or(x, S::template shift_right<6>(x)),
            S::broadcast(0x000F000F000F000FULL << offset)
        );
        x = S::bitwise_and(
            S::bitwise_or(x, S::template shift_right<12>(x)),
            S::broadcast(0x000000FF000000FFULL << offset)
        );
        x = S::bitwise_or(x, S::template shift_right<24>(x));
        if constexpr (offset != 0) {
            x = S::template shift_right<offset>(x);
        }
        return x;
    }
};

// =================================================================================================
//     Word Encoder Direct
// =================================================================================================

/**
 * @brief Encode ASCII characters into packed 2-bit codes via the direct code packing of the arch.
 *
 * Instead of gathering bits under a mask, this reduces each character to its 2-bit code, and then
 * packs each four codes into a byte with a few instructions that are specific to the SIMD arch,
 * see `store_two_bit_codes()` in core/simd.hpp. This is the fastest word encoder on every arch.
 *
 * Where the arch has a byte shuffle, the codes are looked up in a table. For invalid characters
 * of 128 and above, this can give a different code than the other word encoders do.
 *
 * As for WordEncoderButterfly, the call operator on a whole `simd_vector` processes `lanes` words
 * at once, and the one on a single word the rest.
 */
template <Encoding E, Layout L, SimdArch Simd = SimdScalar>
struct WordEncoderDirect
{
    static constexpr Encoding encoding = E;
    static constexpr Layout layout = L;
    static constexpr std::size_t lanes = Simd::lanes;
    using simd_vector = typename Simd::simd_vector;

    static simd_vector load_unaligned(void const* a) noexcept
    {
        return Simd::load_unaligned(a);
    }

    // Each byte holds the code of its base, see the call operator below.
    static void store_packed(simd_vector const& v, char* out) noexcept
    {
        if constexpr (L == Layout::kMSB) {
            Simd::template store_two_bit_codes<true>(v, out);
        } else if constexpr (L == Layout::kLSB) {
            Simd::template store_two_bit_codes<false>(v, out);
        } else {
            static_assert(
                dependent_false_v<L>, "Unhandled Layout in WordEncoderDirect::store_packed()"
            );
        }
    }

    static void store_valid_bits(simd_vector const& v, std::uint8_t* out) noexcept
    {
        store_valid_nucleotide_bits_<Simd>(v, out);
    }

    // For SimdScalar, the vector is a single word, which is encoded by the call operator below.
    FISK_ALWAYS_INLINE
    simd_vector operator()(simd_vector x) const noexcept
    requires (lanes > 1)
    {
        return codes_<Simd>(x);
    }

    FISK_ALWAYS_INLINE
    std::uint64_t operator()(std::uint64_t word) const noexcept
    {
        return SimdScalar::pack_two_bit_codes_(codes_<SimdScalar>(pack_orient_chunk_<L>(word)));
    }

protected:

    // The code of a character, for each value of its low four bits, which are all that the code
    // depends on.
    static constexpr std::array<std::uint8_t, 16> code_table_ = [] {
        std::array<std::uint8_t, 16> table{};
        for (std::uint64_t nibble = 0; nibble < table.size(); ++nibble) {
            std::uint64_t const code = pack_prepare_codes_<E>(nibble) & pack_mask_v<E>;
            table[nibble] = static_cast<std::uint8_t>(code >> std::countr_zero(pack_mask_v<E>));
        }
        return table;
    }();

    // Reduce each byte to its 2-bit code, in its two lowest bits: with a table lookup where the
    // arch has one, and with shifts otherwise. The shifts are per lane, but what they move across
    // byte boundaries is masked out.
    template <SimdArch S>
    FISK_ALWAYS_INLINE
    static typename S::simd_vector codes_(typename S::simd_vector x) noexcept
    {
        if constexpr (SimdArchWithByteShuffle<S>) {
            return S::lookup_low_nibbles(x, code_table_);
        } else {
            if constexpr (E == Encoding::kACGT) {
                x = S::bitwise_xor(S::template shift_right<1>(x), S::template shift_right<2>(x));
            } else if constexpr (E == Encoding::kACTG) {
                x = S::template shift_right<1>(x);
            } else {
                static_assert(dependent_false_v<E>, "Unhandled Encoding in WordEncoderDirect");
            }
            return S::bitwise_and(x, S::broadcast(0x0303030303030303ULL));
        }
    }
};

// -----------------------------------------------------------------------------
//     Word Encoder Direct Double
// -----------------------------------------------------------------------------

/**
 * @brief Same as WordEncoderDirect, but processing two vectors of the SIMD arch per step.
 *
 * Storing the packed bytes of two vectors together takes fewer instructions than storing them one
 * after the other, see the overload of `store_two_bit_codes()` for two vectors in core/simd.hpp.
 * This is only available for the archs that have that overload, see SimdArchWithDoubleStore, and
 * is then the fastest word encoder for long sequences.
 *
 * The `simd_vector` here holds two vectors of the arch, and `lanes` is accordingly twice that of
 * the arch. Sequences shorter than such a pair are packed one word at a time, which makes this
 * slower than WordEncoderDirect for some short sequences.
 */
template <Encoding E, Layout L, SimdArchWithDoubleStore Simd>
struct WordEncoderDirectDouble : WordEncoderDirect<E, L, Simd>
{
    static constexpr std::size_t lanes = 2 * Simd::lanes;

    // Not a std::array, as GCC warns about vector types as template arguments.
    struct simd_vector
    {
        typename Simd::simd_vector v[2];
    };

    static simd_vector load_unaligned(void const* a) noexcept
    {
        return {{
            Simd::load_unaligned(a),
            Simd::load_unaligned(static_cast<char const*>(a) + 8 * Simd::lanes)
        }};
    }

    static void store_packed(simd_vector const& v, char* out) noexcept
    {
        if constexpr (L == Layout::kMSB) {
            Simd::template store_two_bit_codes<true>(v.v[0], v.v[1], out);
        } else if constexpr (L == Layout::kLSB) {
            Simd::template store_two_bit_codes<false>(v.v[0], v.v[1], out);
        } else {
            static_assert(
                dependent_false_v<L>, "Unhandled Layout in WordEncoderDirectDouble::store_packed()"
            );
        }
    }

    static void store_valid_bits(simd_vector const& v, std::uint8_t* out) noexcept
    {
        store_valid_nucleotide_bits_<Simd>(v.v[0], out);
        store_valid_nucleotide_bits_<Simd>(v.v[1], out + Simd::lanes);
    }

    FISK_ALWAYS_INLINE
    simd_vector operator()(simd_vector const& x) const noexcept
    {
        return {{
            WordEncoderDirect<E, L, Simd>::template codes_<Simd>(x.v[0]),
            WordEncoderDirect<E, L, Simd>::template codes_<Simd>(x.v[1])
        }};
    }

    // The call operator on a single word.
    using WordEncoderDirect<E, L, Simd>::operator();
};

} // namespace fisk
