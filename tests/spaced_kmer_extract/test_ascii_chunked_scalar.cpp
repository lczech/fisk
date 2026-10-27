#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/core/types.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"
#include "fisk/spaced_kmer_extract/ascii_chunked_scalar.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"
#include "spaced_kmer_extract/common.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers
// =================================================================================================

// The unchecked or checked function, with the masks of `MaskMaker` and the bit extraction
// `BitExtract`, as an `extract` for the checks of the chunked loops, see
// spaced_kmer_extract/common.hpp.
template <bool Checked, typename MaskMaker, typename BitExtract>
static auto make_extract(MaskMaker make_masks, BitExtract bit_ext)
{
    return [=]<std::size_t ChunkSize, Encoding E, Layout L>(
        std::string const& seq, std::vector<std::string> const& mask_strings, auto single,
        auto* scratch, auto const& callback
    ) {
        WordEncoderButterfly<E, L> const encoder;
        auto const masks = make_masks(make_spaced_kmer_masks<L>(mask_strings));
        using Mask = typename decltype(masks)::value_type;
        auto const run = [&](auto const& mask_or_masks) {
            if constexpr (Checked) {
                if (scratch) {
                    for_each_spaced_kmer_ascii_checked_chunked_scalar<ChunkSize>(
                        seq, mask_or_masks, encoder, bit_ext, *scratch, callback
                    );
                } else {
                    for_each_spaced_kmer_ascii_checked_chunked_scalar<ChunkSize>(
                        seq, mask_or_masks, encoder, bit_ext, callback
                    );
                }
            } else {
                if (scratch) {
                    for_each_spaced_kmer_ascii_unchecked_chunked_scalar<ChunkSize>(
                        seq, mask_or_masks, encoder, bit_ext, *scratch, callback
                    );
                } else {
                    for_each_spaced_kmer_ascii_unchecked_chunked_scalar<ChunkSize>(
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

// The checks of the chunked loops, with the bit extraction of the bit loop and of the butterfly
// network, see bit_extract/bit_extract.hpp.
template <bool Checked>
static void check_chunked()
{
    auto const as_is = [](auto masks) { return masks; };
    auto const butterfly = [](auto masks) { return make_spaced_kmer_butterfly_masks(masks); };
    check_spaced_ascii_chunked_all<Checked, false>(
        make_extract<Checked>(as_is, bit_extract_bit_loop)
    );
    check_spaced_ascii_chunked_all<Checked, false>(
        make_extract<Checked>(butterfly, bit_extract_butterfly)
    );
}

// =================================================================================================
//     for_each_spaced_kmer_ascii_unchecked_chunked_scalar()
// =================================================================================================

TEST(SpacedKmerExtractAsciiChunkedScalar, Unchecked)
{
    check_chunked<false>();
}

// =================================================================================================
//     for_each_spaced_kmer_ascii_checked_chunked_scalar()
// =================================================================================================

TEST(SpacedKmerExtractAsciiChunkedScalar, Checked)
{
    check_chunked<true>();
}
