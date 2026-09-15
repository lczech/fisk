#pragma once

#include <cstdint>
#include <cstring>
#include <string_view>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/types.hpp"
#include "fisk/bit_extract/bit_extract.hpp"

// =================================================================================================
//     8-Base Extractors -- PEXT
// =================================================================================================

// Each of these encodes 8 ASCII characters (one uint64_t, little-endian-loaded via memcpy) into
// their packed 2-bit codes in a single PEXT call, and is usable directly as the `Extractor`
// argument to pack_sequence() below. Each carries its own `encoding` and `layout` tags as
// `static constexpr` members right next to the logic that implements them, so the two cannot
// silently drift apart.
//
// The `Msb`-suffixed variants prepend a byte_swap_64() before extracting, so that the first
// character of each 8-byte chunk lands in the high bits of the result instead of the low bits;
// see Layout in core/types.hpp for details.

#if defined(FISK_HAS_BMI2)

/**
 * @brief Encode 8 ASCII ACTG characters via a single direct PEXT call on the raw bytes (see
 * char_to_nt_ascii_actg()), LSB-first: the chunk's first character lands in the low 2 bits.
 */
struct EncodeActg8PextLsb
{
    static constexpr Encoding encoding = Encoding::kACTG;
    static constexpr Layout layout = Layout::kLSB;

    inline std::uint64_t operator()(std::uint64_t word) const noexcept
    {
        return bit_extract_pext(word, BitExtractMask(0x0606060606060606ULL));
    }
};

/**
 * @brief Same as EncodeActg8PextLsb, but MSB-first.
 */
struct EncodeActg8PextMsb
{
    static constexpr Encoding encoding = Encoding::kACTG;
    static constexpr Layout layout = Layout::kMSB;

    inline std::uint64_t operator()(std::uint64_t word) const noexcept
    {
        return bit_extract_pext(byte_swap_64(word), BitExtractMask(0x0606060606060606ULL));
    }
};

/**
 * @brief Encode 8 ASCII ACGT characters via the same SWAR bit-trick as char_to_nt_ascii_acgt(),
 * applied to all 8 bytes at once, followed by a single PEXT call to pack the codes densely.
 * LSB-first: the chunk's first character lands in the low 2 bits.
 */
struct EncodeAcgt8PextLsb
{
    static constexpr Encoding encoding = Encoding::kACGT;
    static constexpr Layout layout = Layout::kLSB;

    inline std::uint64_t operator()(std::uint64_t word) const noexcept
    {
        std::uint64_t const acgt = ((word >> 1) ^ (word >> 2)) & 0x0303030303030303ULL;
        return bit_extract_pext(acgt, BitExtractMask(0x0303030303030303ULL));
    }
};

/**
 * @brief Same as EncodeAcgt8PextLsb, but MSB-first.
 */
struct EncodeAcgt8PextMsb
{
    static constexpr Encoding encoding = Encoding::kACGT;
    static constexpr Layout layout = Layout::kMSB;

    inline std::uint64_t operator()(std::uint64_t word) const noexcept
    {
        std::uint64_t const swapped = byte_swap_64(word);
        std::uint64_t const acgt = ((swapped >> 1) ^ (swapped >> 2)) & 0x0303030303030303ULL;
        return bit_extract_pext(acgt, BitExtractMask(0x0303030303030303ULL));
    }
};

#endif // FISK_HAS_BMI2

// =================================================================================================
//     8-Base Extractors -- Butterfly Table
// =================================================================================================

// Same four combinations as above, but using the portable software butterfly-network bit
// extraction instead of hardware PEXT: the fallback tier for hardware without fast PEXT (e.g.
// pre-Zen3 AMD, where PEXT/PDEP are microcoded and slow), or without PEXT at all.
//
// Each mask is fixed for a given encoding, so the corresponding BitExtractButterflyTable
// is precomputed once via a function-local static.

/**
 * @brief Same as EncodeActg8PextLsb, via the software butterfly network instead of PEXT.
 */
struct EncodeActg8ButterflyLsb
{
    static constexpr Encoding encoding = Encoding::kACTG;
    static constexpr Layout layout = Layout::kLSB;

    inline std::uint64_t operator()(std::uint64_t word) const noexcept
    {
        static const BitExtractButterflyTable table =
            bit_extract_butterfly_table_preprocess(0x0606060606060606ULL);
        return bit_extract_butterfly_table(word, table);
    }
};

/**
 * @brief Same as EncodeActg8PextMsb, via the software butterfly network instead of PEXT.
 */
struct EncodeActg8ButterflyMsb
{
    static constexpr Encoding encoding = Encoding::kACTG;
    static constexpr Layout layout = Layout::kMSB;

    inline std::uint64_t operator()(std::uint64_t word) const noexcept
    {
        static const BitExtractButterflyTable table =
            bit_extract_butterfly_table_preprocess(0x0606060606060606ULL);
        return bit_extract_butterfly_table(byte_swap_64(word), table);
    }
};

/**
 * @brief Same as EncodeAcgt8PextLsb, via the software butterfly network instead of PEXT.
 */
struct EncodeAcgt8ButterflyLsb
{
    static constexpr Encoding encoding = Encoding::kACGT;
    static constexpr Layout layout = Layout::kLSB;

    inline std::uint64_t operator()(std::uint64_t word) const noexcept
    {
        static const BitExtractButterflyTable table =
            bit_extract_butterfly_table_preprocess(0x0303030303030303ULL);
        std::uint64_t const acgt = ((word >> 1) ^ (word >> 2)) & 0x0303030303030303ULL;
        return bit_extract_butterfly_table(acgt, table);
    }
};

/**
 * @brief Same as EncodeAcgt8PextMsb, via the software butterfly network instead of PEXT.
 */
struct EncodeAcgt8ButterflyMsb
{
    static constexpr Encoding encoding = Encoding::kACGT;
    static constexpr Layout layout = Layout::kMSB;

    inline std::uint64_t operator()(std::uint64_t word) const noexcept
    {
        static const BitExtractButterflyTable table =
            bit_extract_butterfly_table_preprocess(0x0303030303030303ULL);
        std::uint64_t const swapped = byte_swap_64(word);
        std::uint64_t const acgt = ((swapped >> 1) ^ (swapped >> 2)) & 0x0303030303030303ULL;
        return bit_extract_butterfly_table(acgt, table);
    }
};

// =================================================================================================
//     Chunk Write Helper
// =================================================================================================

/**
 * @brief Write the packed bases of one extractor call to `dest`, respecting `L`.
 */
template <Layout L, int Bytes = 2>
inline void write_two_bit_chunk(char* dest, std::uint64_t value) noexcept
{
    // Each extractor above packs 8 bases (16 bits) into the low bits of its return value. A byte is
    // the smallest addressable unit of PackedSequence's storage, so those 16 bits become 2
    // sequential output bytes. For `kLSB`, the extractor already produces those 2 bytes in the
    // right relative order for free (the first of the 8 bases ends up in the low bits of the first
    // byte, and so on); for `kMSB`, the two bytes come out address-reversed relative to base order
    // (the byte holding the later 4 bases lands first), so they need a fix-up swap before the
    // sequential write.
    //
    // `Bytes` (1 or 2) is a template parameter rather than a runtime one so that the hot full-chunk
    // call sites below (always 2) keep a fixed-width store with no added branch; only the ragged tail
    // chunk (<=4 real bases) ever instantiates the 1-byte write, and byte 0 -- the first of the two
    // output bytes post-swap -- is the one holding those bases in both bit orders.

    static_assert(Bytes == 1 || Bytes == 2, "write_two_bit_chunk() only supports 1 or 2 bytes");
    assert(dest != nullptr);
    assert(value <= 0xFFFF);
    std::uint16_t v16 = static_cast<std::uint16_t>(value);
    if constexpr (L == Layout::kMSB) {
        v16 = byte_swap_16(v16);
    }
    std::memcpy(dest, &v16, Bytes);
}

// =================================================================================================
//     pack_sequence()
// =================================================================================================

/**
 * @brief Pack a whole ASCII sequence into a PackedSequence, reusing existing storage.
 *
 * Processes the sequence in 8-byte chunks via `extract` (see the extractor structs above), writing
 * each chunk's 16-bit result to its 2 sequential output bytes at `out.data[i/4 .. i/4+1]`, where
 * `i` is the chunk's starting byte offset into `seq` (see write_two_bit_chunk() for the per-byte
 * ordering).
 *
 * `Extractor::encoding` and `Extractor::layout` (see Encoding and Layout in core/types.hpp) select
 * the out type. Passing an extractor whose encoding or layout differs from `out`'s is a compile
 * error.
 */
template <typename Extractor>
inline void pack_sequence(
    std::string_view seq,
    Extractor const& extract,
    PackedSequence<Extractor::encoding, Extractor::layout>& out
) {
    constexpr Layout layout = Extractor::layout;

    std::size_t const seq_len   = seq.size();
    std::size_t const num_bytes = (seq_len + 3) / 4;

    out.length = seq_len;
    out.data.assign(num_bytes, 0);
    char* const out_bytes = reinterpret_cast<char*>(out.data.data());
    char const* const data = seq.data();

    // Read the 8-byte chunk starting at seq byte offset `off`.
    auto read_chunk = [data](std::size_t off) -> std::uint64_t {
        std::uint64_t word;
        std::memcpy(&word, data + off, 8);
        return word;
    };

    // byte offset into seq
    std::size_t i = 0;

    // Full 8-base chunks.
    for (; i + 8 <= seq_len; i += 8) {
        write_two_bit_chunk<layout>(out_bytes + i / 4, extract(read_chunk(i)));
    }

    // Remaining < 8 bases: zero-pad into a local word before extracting. `out` holds no trailing
    // bytes beyond its real content, so a <=4-base remainder (one real output byte, not two) must
    // write only that one byte -- writing the usual 2 would spill past the allocation.
    if (i < seq_len) {
        std::size_t const remaining = seq_len - i;
        std::uint64_t word = 0;
        std::memcpy(&word, data + i, remaining);
        if (remaining <= 4) {
            write_two_bit_chunk<layout, 1>(out_bytes + i / 4, extract(word));
        } else {
            write_two_bit_chunk<layout, 2>(out_bytes + i / 4, extract(word));
        }
    }
}

/**
 * @brief Pack a whole ASCII sequence into a freshly allocated PackedSequence.
 *
 * Convenience wrapper around the in-place overload above. Prefer that overload directly when
 * packing many sequences in a loop (e.g. in a benchmark), reusing one PackedSequence across
 * calls, to avoid attributing repeated heap allocation to whatever is being measured.
 */
template <typename Extractor>
inline PackedSequence<Extractor::encoding, Extractor::layout> pack_sequence(
    std::string_view seq, Extractor const& extract
) {
    PackedSequence<Extractor::encoding, Extractor::layout> out;
    pack_sequence(seq, extract, out);
    return out;
}
