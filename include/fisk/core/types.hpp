#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// =================================================================================================
//     Conventions
// =================================================================================================

// Packed 2-bit data can differ along two independent axes: which base each 2-bit code stands for
// (Encoding), and in which order bases are laid out within a word (Layout). Both are
// tracked as template parameters on the types that carry such data, so that the compiler can
// enforce consistency between etween producer and consumer.

/**
 * @brief Encoding for which base each 2-bit code stands for.
 *
 * `kACGT`: A=0, C=1, G=2, T=3 (alphabetical). Complementary bases are bitwise complements of each
 * other, which makes reverse-complement handling cheap. Use this if you need to compute reverse
 * complements, or if you need lexicographic string order to match integer order.
 *
 * `kACTG`: A=0, C=1, T=2, G=3. Falls out directly of bits 1-2 of the ASCII code, making it cheaper
 * to encode, at the cost of the cheap-complement property. Use this if performance matters, but
 * lexicographically ordered encoding and reverse-complement handling are not required.
 */
enum class Encoding
{
    kACGT,
    kACTG
};

/**
 * @brief Layout in which order bases are stored within the bits of a packed word.
 *
 * `kMSB`: earlier bases of the input sequence occupy the most significant bits, matching the
 * left-rolling k-mer convention `kmer = (kmer << 2) | code`, and hence preserving lexicographic
 * string order as integer order.
 *
 * `kLSB`: earlier bases of the input sequence occupy the least significant bits, matching the
 * right-rolling k-mer convention `kmer = (kmer >> 2) | (code << (2 * (k - 1)))`. This is the order
 * that falls out directly of a little-endian bit extraction via batch encode of 8 bytes treated
 * as one 64 bit word with no extra transformation, and is therefore cheaper to produce with PEXT,
 * at the cost of not preserving lexicographic order.
 */
enum class Layout
{
    kMSB,
    kLSB
};

// =================================================================================================
//     Packed Sequence
// =================================================================================================

/**
 * @brief A nucleotide sequence, densely packed at 2 bits per base.
 *
 * `data` holds exactly `ceil(length / 4)` bytes of 4 bases each, with no trailing padding.
 * Readers are responsible for bounds checking when reading from `data`.
 *
 * `E` states which base each 2-bit code stands for (see Encoding), and `L` how the 4 bases are
 * arranged within each byte (see Layout). Both are also exposed as the `encoding` and `layout`
 * static members.
 */
template <Encoding E, Layout L>
struct PackedSequence
{
    static constexpr Encoding encoding = E;
    static constexpr Layout layout = L;

    std::vector<std::uint8_t> data;
    std::size_t length = 0;
};
