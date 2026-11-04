#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/core/types.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"
#include "fisk/spaced_kmer_extract/ascii_chunked_scalar.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"
#include "spaced_kmer_extract/common.hpp"
#include "testing.hpp"

// =================================================================================================
//     Spaced K-mer Extraction from ASCII, Chunked, Scalar: Shared Test Helpers
// =================================================================================================

// The checks of the tests of ascii_chunked_scalar.hpp, which are split into
// test_ascii_chunked_scalar_unchecked.cpp and test_ascii_chunked_scalar_checked.cpp, so that each
// compiles with less memory.

// The unchecked or checked function, with the masks of `MaskMaker` and the bit extraction
// `BitExtract`, as an `extract` for the checks of the chunked loops, see
// spaced_kmer_extract/common.hpp.
template <bool Checked, typename MaskMaker, typename BitExtract>
inline auto make_extract(MaskMaker make_masks, BitExtract bit_ext)
{
    return [=]<std::size_t ChunkSize, fisk::Encoding E, fisk::Layout L>(
        std::string const& seq, std::vector<std::string> const& mask_strings, auto single,
        auto* scratch, auto const& callback
    ) {
        fisk::WordEncoderButterfly<E, L> const encoder;
        auto const masks = make_masks(fisk::make_spaced_kmer_masks<L, 128>(mask_strings));
        using Mask = typename decltype(masks)::value_type;
        auto const run = [&](auto const& mask_or_masks) {
            if constexpr (Checked) {
                if (scratch) {
                    fisk::for_each_spaced_kmer_ascii_checked_chunked_scalar<ChunkSize>(
                        seq, mask_or_masks, encoder, bit_ext, *scratch, callback
                    );
                } else {
                    fisk::for_each_spaced_kmer_ascii_checked_chunked_scalar<ChunkSize>(
                        seq, mask_or_masks, encoder, bit_ext, callback
                    );
                }
            } else {
                if (scratch) {
                    fisk::for_each_spaced_kmer_ascii_unchecked_chunked_scalar<ChunkSize>(
                        seq, mask_or_masks, encoder, bit_ext, *scratch, callback
                    );
                } else {
                    fisk::for_each_spaced_kmer_ascii_unchecked_chunked_scalar<ChunkSize>(
                        seq, mask_or_masks, encoder, bit_ext, callback
                    );
                }
            }
        };
        if constexpr (decltype(single)::value) {
            run(masks.front());
        } else if (masks.size() == 1) {
            run(std::span<Mask const, 1>(masks.data(), 1));
        } else {
            run(std::span<Mask const, 3>(masks.data(), 3));
        }
    };
}

// The checks of the chunked loops, with the bit extraction of the bit loop, of the butterfly
// network, and of PEXT, see bit_extract/bit_extract.hpp, for spans up to 32 and above.
template <bool Checked>
inline void check_chunked()
{
    auto const as_is = [](auto masks) { return masks; };
    auto const butterfly = [](auto masks) { return fisk::make_spaced_kmer_butterfly_masks(masks); };
    for (auto const* mask_sets : {&chunked_mask_sets(), &chunked_wide_mask_sets()}) {
        check_spaced_ascii_chunked_all<Checked, false>(
            make_extract<Checked>(as_is, fisk::bit_extract_bit_loop), *mask_sets
        );
        check_spaced_ascii_chunked_all<Checked, false>(
            make_extract<Checked>(butterfly, fisk::bit_extract_butterfly), *mask_sets
        );
        #if defined(FISK_HAS_BMI2)
        check_spaced_ascii_chunked_all<Checked, false>(
            make_extract<Checked>(as_is, fisk::bit_extract_pext), *mask_sets
        );
        #endif
    }
}

// The checks of check_spaced_ascii_chunked_error_contract() (spaced_kmer_extract/common.hpp), with
// the bit extraction of the bit loop.
template <bool Checked, std::size_t MaxSpan>
inline void check_chunked_error_contract()
{
    constexpr fisk::Encoding E = fisk::Encoding::kACGT;
    constexpr fisk::Layout L = fisk::Layout::kMSB;
    using Scratch = std::conditional_t<
        Checked, fisk::PackedSequenceWithValidity<E, L>, fisk::PackedSequence<E, L>
    >;
    fisk::WordEncoderButterfly<E, L> const encoder;
    Scratch scratch;
    auto const noop = [](std::size_t, std::size_t, auto const&) {};
    auto const as_is = [](auto masks) { return masks; };
    check_spaced_ascii_chunked_error_contract<MaxSpan>(as_is, [&](
        std::string const& seq, auto const& mask_or_masks
    ) {
        if constexpr (Checked) {
            EXPECT_THROW(
                fisk::for_each_spaced_kmer_ascii_checked_chunked_scalar<7>(
                    seq, mask_or_masks, encoder, fisk::bit_extract_bit_loop, scratch, noop
                ),
                std::invalid_argument
            );
            EXPECT_THROW(
                fisk::for_each_spaced_kmer_ascii_checked_chunked_scalar<7>(
                    seq, mask_or_masks, encoder, fisk::bit_extract_bit_loop, noop
                ),
                std::invalid_argument
            );
        } else {
            EXPECT_THROW(
                fisk::for_each_spaced_kmer_ascii_unchecked_chunked_scalar<7>(
                    seq, mask_or_masks, encoder, fisk::bit_extract_bit_loop, scratch, noop
                ),
                std::invalid_argument
            );
            EXPECT_THROW(
                fisk::for_each_spaced_kmer_ascii_unchecked_chunked_scalar<7>(
                    seq, mask_or_masks, encoder, fisk::bit_extract_bit_loop, noop
                ),
                std::invalid_argument
            );
        }
    });
}
