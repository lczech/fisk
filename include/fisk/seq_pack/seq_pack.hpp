#pragma once

#include <cstdint>
#include <cstring>
#include <string_view>
#include <type_traits>
#include <utility>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/seq_enc.hpp"
#include "fisk/bit_extract/bit_extract.hpp"

// =================================================================================================
//     8-Base Extractors -- PEXT
// =================================================================================================

// Each of these encodes 8 ASCII characters (one uint64_t, little-endian-loaded via memcpy) into
// their packed 2-bit codes in a single PEXT call, and is usable directly as the `Extractor`
// argument to pack_sequence() below. Each carries its own `order` as a `static constexpr` member
// right next to the logic that implements it, so the two cannot silently drift apart.
//
// The `Msb`-suffixed variants prepend a byte_swap_64() before extracting, so that the first
// character of each 8-byte chunk lands in the high bits of the result instead of the low bits;
// see BitOrder in core/seq_enc.hpp for details.

#if defined(FISK_HAS_BMI2)

/**
 * @brief Encode 8 ASCII ACTG characters via a single direct PEXT call on the raw bytes (see
 * char_to_nt_ascii_actg()), LSB-first: the chunk's first character lands in the low 2 bits.
 */
struct EncodeActg8PextLsb
{
    static constexpr BitOrder order = BitOrder::Lsb;

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
    static constexpr BitOrder order = BitOrder::Msb;

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
    static constexpr BitOrder order = BitOrder::Lsb;

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
    static constexpr BitOrder order = BitOrder::Msb;

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
// Each mask is fixed for a given (convention) pair, so the corresponding BitExtractButterflyTable
// is precomputed once via a function-local static.

/**
 * @brief Same as EncodeActg8PextLsb, via the software butterfly network instead of PEXT.
 */
struct EncodeActg8ButterflyLsb
{
    static constexpr BitOrder order = BitOrder::Lsb;

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
    static constexpr BitOrder order = BitOrder::Msb;

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
    static constexpr BitOrder order = BitOrder::Lsb;

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
    static constexpr BitOrder order = BitOrder::Msb;

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
//     pack_sequence()
// =================================================================================================

/**
 * @brief Pack a whole ASCII sequence into a TwoBitSequence, reusing existing storage.
 *
 * Processes the sequence in 8-byte chunks via `extract` (see the extractor structs above),
 * writing each chunk's 16-bit result directly to its byte offset in `out.data` (offset `2k` for
 * `Lsb`, `6-2k` for `Msb`, where `k` is the chunk's index 0..3 within its 32-base word).
 *
 * `Extractor::order` (see BitOrder in core/seq_enc.hpp) both selects the out type. Passing an
 * extractor of the wrong order for a given `out` is a compile error.
 */
template <typename Extractor>
inline void pack_sequence(
    std::string_view seq,
    Extractor&& extract,
    TwoBitSequence<std::remove_cvref_t<Extractor>::order>& out
) {
    // Extractor deduces to a reference type whenever the caller passes a named extractor object
    // rather than a temporary; strip that off before looking up `order`,
    // since a qualified-id lookup does not see through a reference type on its own.
    constexpr BitOrder order = std::remove_cvref_t<Extractor>::order;

    std::size_t const seq_len   = seq.size();
    std::size_t const num_words = (seq_len + 31) / 32;

    out.length = seq_len;
    out.data.assign(num_words + 1, 0); // +1 trailing all-zero sentinel word
    char* const out_bytes = reinterpret_cast<char*>(out.data.data());

    // Byte offset of chunk k (0..3) within its own 8-byte output word.
    auto chunk_offset = [](std::size_t k) {
        return order == BitOrder::Msb ? (6 - 2 * k) : (2 * k);
    };

    // Write `value`'s low 16 bits to `dest`.
    auto write_chunk = [](char* dest, std::uint64_t value) {
        std::uint16_t const v16 = static_cast<std::uint16_t>(value);
        std::memcpy(dest, &v16, 2);
    };

    // Read the 8-byte chunk starting at seq byte offset `off`.
    auto read_chunk = [data = seq.data()](std::size_t off) -> std::uint64_t {
        std::uint64_t word;
        std::memcpy(&word, data + off, 8);
        return word;
    };

    char const* const data = seq.data();
    std::size_t i = 0; // byte offset into seq
    std::size_t word_idx = 0;

    // Full 32-byte (32-base) blocks: 4 chunks of 8 bytes each per output word,
    // written into one 64-bit word.
    for (; i + 32 <= seq_len; i += 32, ++word_idx) {
        char* const word_ptr = out_bytes + word_idx * 8;
        write_chunk(word_ptr + chunk_offset(0), extract(read_chunk(i +  0)));
        write_chunk(word_ptr + chunk_offset(1), extract(read_chunk(i +  8)));
        write_chunk(word_ptr + chunk_offset(2), extract(read_chunk(i + 16)));
        write_chunk(word_ptr + chunk_offset(3), extract(read_chunk(i + 24)));
    }

    // Remaining < 32 bases: write only the chunks that actually exist.
    char* const word_ptr = out_bytes + word_idx * 8;
    for (std::size_t k = 0; i < seq_len && k < 4; ++k) {
        std::size_t const remaining = seq_len - i;
        std::size_t const take = remaining < 8 ? remaining : std::size_t{8};

        std::uint64_t word = 0;
        std::memcpy(&word, data + i, take);

        write_chunk(word_ptr + chunk_offset(k), extract(word));
        i += take;
    }
}

/**
 * @brief Pack a whole ASCII sequence into a freshly allocated TwoBitSequence.
 *
 * Convenience wrapper around the in-place overload above. Prefer that overload directly when
 * packing many sequences in a loop (e.g. in a benchmark), reusing one TwoBitSequence across
 * calls, to avoid attributing repeated heap allocation to whatever is being measured.
 */
template <typename Extractor>
inline TwoBitSequence<std::remove_cvref_t<Extractor>::order> pack_sequence(
    std::string_view seq, Extractor&& extract
) {
    TwoBitSequence<std::remove_cvref_t<Extractor>::order> out;
    pack_sequence(seq, std::forward<Extractor>(extract), out);
    return out;
}
