#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "fisk/core/char_encoder.hpp"
#include "fisk/core/types.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "testing.hpp"

// =================================================================================================
//     Helpers and Oracle
// =================================================================================================

// Ground truth two-bit codes come from oracle_code() in oracle.hpp, independent of the encoders
// under test.

// Decode base `i` per PackedSequence's documented contract (see Layout in core/types.hpp),
// independent of pack_sequence_unchecked()'s internals.
template <fisk::PackedSequenceType Seq>
inline int decode_base(Seq const& s, std::size_t i)
{
    std::uint8_t const byte = s.codes()[i / 4];
    std::size_t const in_byte = i % 4;
    if constexpr (Seq::layout == fisk::Layout::kMSB) {
        return (byte >> (6 - 2 * in_byte)) & 0x3;
    } else {
        return (byte >> (2 * in_byte)) & 0x3;
    }
}

// Checks a packed sequence against `seq` and the oracle: length, number of bytes of codes, the
// code of every valid base, and zeros beyond the length.
template <fisk::PackedSequenceType Seq>
inline void check_packed(std::string const& seq, Seq const& packed)
{
    EXPECT_EQ(packed.length(), seq.size());
    EXPECT_EQ(packed.codes().size(), (seq.size() + 3) / 4);

    for (std::size_t i = 0; i < seq.size(); ++i) {
        std::uint8_t const code = oracle_code<Seq::encoding>(seq[i]);
        if (code < fisk::kInvalidNucleotide) {
            EXPECT_EQ(decode_base(packed, i), static_cast<int>(code));
        }
    }
    check_zero_beyond_length(packed);
}

// -------------------------------------------------------------------------------------------------
//     Validity
// -------------------------------------------------------------------------------------------------

inline bool is_valid_nucleotide(char c)
{
    return kValidBases.find(c) != std::string_view::npos;
}
