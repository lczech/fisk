#include <concepts>
#include <cstddef>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "fisk/bit_extract/simd.hpp"
#include "fisk/core/cast.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/simd/avx2.hpp"
#include "fisk/simd/avx512.hpp"
#include "fisk/simd/neon.hpp"
#include "fisk/simd/scalar.hpp"
#include "fisk/simd/sse2.hpp"
#include "fisk/core/types.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"
#include "fisk/spaced_kmer_extract/ascii_chunked_simd.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"
#include "spaced_kmer_extract/common.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers
// =================================================================================================

// Vector callbacks that hand on the spaced k-mers of the valid lanes to a callback of single
// spaced k-mers, with for_each_valid_lane(): with mask_idx, and, for a single kernel, without it.
// Named types, for the reason given in spaced_kmer_extract/common.hpp.
template <typename Callback>
struct SpacedLaneCallback
{
    Callback const& callback;

    template <typename V, typename Valid>
    void operator()(std::size_t pos, std::size_t mask_idx, V spaced_kmers, Valid valid) const
    {
        for_each_valid_lane(pos, spaced_kmers, valid, [&](std::size_t p, auto const& spaced_kmer) {
            callback(p, mask_idx, spaced_kmer);
        });
    }
};

template <typename Callback>
struct SpacedLaneShorthandCallback
{
    Callback const& callback;

    template <typename V, typename Valid>
    void operator()(std::size_t pos, V spaced_kmers, Valid valid) const
    {
        for_each_valid_lane(pos, spaced_kmers, valid, callback);
    }
};

// The unchecked or checked function, with the kernels `Kernel`, as an `extract` for the checks of
// the chunked loops, see spaced_kmer_extract/common.hpp. A callback with mask_idx gets its spaced
// k-mers from the vector form with mask_idx, the others from the vector shorthand without it.
template <bool Checked, typename Kernel>
static constexpr auto extract = []<std::size_t ChunkSize, Encoding E, Layout L>(
    std::string const& seq, std::vector<std::string> const& mask_strings, auto single,
    auto* scratch, auto const& callback
) {
    using Callback = std::remove_cvref_t<decltype(callback)>;
    using Mask = SpacedKmerMask<L, Kernel>;
    WordEncoderButterfly<E, L> const encoder;
    auto const kernels = make_spaced_kmer_kernels<Kernel>(make_spaced_kmer_masks<L>(mask_strings));
    auto const run = [&](auto const& kernel_or_kernels, auto const& vector_callback) {
        if constexpr (Checked) {
            if (scratch) {
                for_each_spaced_kmer_ascii_checked_chunked_simd<ChunkSize>(
                    seq, kernel_or_kernels, encoder, *scratch, vector_callback
                );
            } else {
                for_each_spaced_kmer_ascii_checked_chunked_simd<ChunkSize>(
                    seq, kernel_or_kernels, encoder, vector_callback
                );
            }
        } else {
            if (scratch) {
                for_each_spaced_kmer_ascii_unchecked_chunked_simd<ChunkSize>(
                    seq, kernel_or_kernels, encoder, *scratch, vector_callback
                );
            } else {
                for_each_spaced_kmer_ascii_unchecked_chunked_simd<ChunkSize>(
                    seq, kernel_or_kernels, encoder, vector_callback
                );
            }
        }
    };
    if constexpr (decltype(single)::value) {
        if constexpr (std::invocable<Callback, std::size_t, std::size_t, SpacedKmer<E, L>>) {
            run(kernels.front(), SpacedLaneCallback<Callback>{callback});
        } else {
            run(kernels.front(), SpacedLaneShorthandCallback<Callback>{callback});
        }
    } else if (kernels.size() == 1) {
        run(std::span<Mask const, 1>(kernels.data(), 1), SpacedLaneCallback<Callback>{callback});
    } else {
        run(std::span<Mask const, 3>(kernels.data(), 3), SpacedLaneCallback<Callback>{callback});
    }
};

// Both functions with the butterfly kernel of the arch `Simd`.
template <SimdArch Simd>
static void check_chunked_simd()
{
    using Kernel = BitExtractKernelButterfly<Simd>;
    check_spaced_ascii_chunked_all<false, true>(extract<false, Kernel>);
    check_spaced_ascii_chunked_all<true, true>(extract<true, Kernel>);
}

// =================================================================================================
//     for_each_spaced_kmer_ascii_unchecked_chunked_simd(),
//     for_each_spaced_kmer_ascii_checked_chunked_simd()
// =================================================================================================

TEST(SpacedKmerExtractAsciiChunkedSimd, Scalar)
{
    check_chunked_simd<SimdScalar>();
}

#if defined(FISK_HAS_SSE2)
TEST(SpacedKmerExtractAsciiChunkedSimd, Sse2)
{
    check_chunked_simd<SimdSSE2>();
}
#endif

#if defined(FISK_HAS_AVX2)
TEST(SpacedKmerExtractAsciiChunkedSimd, Avx2)
{
    check_chunked_simd<SimdAVX2>();
}
#endif

#if defined(FISK_HAS_AVX512)
TEST(SpacedKmerExtractAsciiChunkedSimd, Avx512)
{
    check_chunked_simd<SimdAVX512>();
}
#endif

#if defined(FISK_HAS_NEON)
TEST(SpacedKmerExtractAsciiChunkedSimd, Neon)
{
    check_chunked_simd<SimdNEON>();
}
#endif
