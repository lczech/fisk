#include <algorithm>
#include <cstddef>
#include <string>
#include <type_traits>

#include "fisk/core/cast.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/simd/simd.hpp"
#include "fisk/simd/avx2.hpp"
#include "fisk/simd/avx512.hpp"
#include "fisk/simd/neon.hpp"
#include "fisk/simd/scalar.hpp"
#include "fisk/simd/sse2.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/ascii_chunked_simd.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"
#include "kmer_extract/common.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     for_each_kmer_ascii_unchecked_chunked_simd(), for_each_kmer_ascii_checked_chunked_simd()
// =================================================================================================

// A vector callback that hands on the k-mers of the valid lanes to a callback of single k-mers,
// with for_each_valid_lane(). A named type, for the reason given in kmer_extract/common.hpp.
template <typename Callback>
struct LaneCallback
{
    Callback const& callback;

    template <typename Kmers, typename Valid>
    void operator()(std::size_t pos, Kmers kmers, Valid valid) const
    {
        for_each_valid_lane(pos, kmers, valid, callback);
    }
};

// The function with the arch `Simd` as an `extract` for the checks of the chunked loops, see
// kmer_extract/common.hpp.
template <SimdArch Simd>
static constexpr auto unchecked = []<std::size_t ChunkSize>(
    std::string const& seq, std::size_t k, auto const& encoder, auto* scratch,
    auto const& callback
) {
    LaneCallback<std::remove_cvref_t<decltype(callback)>> const lanes{callback};
    if (scratch) {
        for_each_kmer_ascii_unchecked_chunked_simd<Simd, ChunkSize>(
            seq, k, encoder, *scratch, lanes
        );
    } else {
        for_each_kmer_ascii_unchecked_chunked_simd<Simd, ChunkSize>(seq, k, encoder, lanes);
    }
};

// The checked function, as above.
template <SimdArch Simd>
static constexpr auto checked = []<std::size_t ChunkSize>(
    std::string const& seq, std::size_t k, auto const& encoder, auto* scratch,
    auto const& callback
) {
    LaneCallback<std::remove_cvref_t<decltype(callback)>> const lanes{callback};
    if (scratch) {
        for_each_kmer_ascii_checked_chunked_simd<Simd, ChunkSize>(
            seq, k, encoder, *scratch, lanes
        );
    } else {
        for_each_kmer_ascii_checked_chunked_simd<Simd, ChunkSize>(seq, k, encoder, lanes);
    }
};

// Both functions with the arch `Simd`: the checks of the chunked loops, and for the checked one
// also those of the checked extraction from ASCII, with chunks of 7 bases.
template <SimdArch Simd>
static void check_chunked_simd()
{
    check_ascii_chunked_all<false>(unchecked<Simd>);
    check_ascii_chunked_all<true>(checked<Simd>);

    auto const extract = ascii_chunked_as_ascii_extractor(checked<Simd>);
    check_ascii_oracle(extract);
    check_ascii_length_boundaries(extract);
    check_ascii_invalid_character_placement(extract);
    check_ascii_invalid_injection_fuzz(extract);
    check_ascii_block_boundaries(extract);
}

// The vectors of the chunked loops, on sequences of valid nucleotides, for every chunk size of
// for_each_chunk_size(). The callbacks are named types, with the chunk size as a member, for the
// reason given in kmer_extract/common.hpp.
//
// Unchecked: the vectors follow each other without gaps, each with 1 to `lanes` k-mers. The last
// one of each chunk may be partial; with a chunk size that is a multiple of 32, each chunk ends on
// a whole vector, so that only the last one of the sequence is partial.
template <SimdArch Simd>
struct UncheckedVectorShape
{
    std::size_t chunk_size;
    std::size_t num_kmers;
    std::size_t* next_pos;

    template <typename Kmers>
    void operator()(std::size_t pos, Kmers, std::size_t valid_count) const
    {
        EXPECT_EQ(pos, *next_pos);
        EXPECT_TRUE(valid_count >= 1 && valid_count <= Simd::lanes);
        if (chunk_size % 32 == 0) {
            std::size_t const expected_count = std::min(Simd::lanes, num_kmers - pos);
            EXPECT_EQ(valid_count, expected_count);
        }
        *next_pos = pos + valid_count;
    }
};

// Checked: each valid k-mer comes once, in order, from a vector that starts in the same chunk as
// the k-mer, at a multiple of `lanes` from the start of the chunk. The vectors after the last
// k-mer of a chunk, up to the end of the last step of the packed loop, have no valid lanes.
template <SimdArch Simd>
struct CheckedVectorShape
{
    std::size_t chunk_size;
    std::size_t* valid;

    template <typename Kmers>
    void operator()(std::size_t pos, Kmers, SimdVectorValidLanes<Simd> valid_lanes) const
    {
        for (std::size_t j = 0; j < Simd::lanes; ++j) {
            if (((valid_lanes.bits >> j) & 1) == 0) {
                continue;
            }
            std::size_t const kmer_pos = pos + j;
            std::size_t const chunk_start = kmer_pos / chunk_size * chunk_size;
            EXPECT_EQ(kmer_pos, *valid);
            EXPECT_TRUE(pos >= chunk_start);
            EXPECT_EQ((pos - chunk_start) % Simd::lanes, std::size_t{0});
            ++*valid;
        }
    }
};

template <SimdArch Simd>
static void check_chunked_simd_vectors()
{
    constexpr Encoding E = Encoding::kACGT;
    constexpr Layout L = Layout::kMSB;
    WordEncoderButterfly<E, L> const encoder;
    for (auto const& seq : chunked_test_sequences()) {
        for (std::size_t const k : {std::size_t{1}, std::size_t{15}, std::size_t{32}}) {
            std::size_t const num_kmers = seq.size() >= k ? seq.size() - k + 1 : 0;
            for_each_chunk_size([&]<std::size_t ChunkSize>() {
                std::size_t next_pos = 0;
                for_each_kmer_ascii_unchecked_chunked_simd<Simd, ChunkSize>(
                    seq, k, encoder, UncheckedVectorShape<Simd>{ChunkSize, num_kmers, &next_pos}
                );
                EXPECT_EQ(next_pos, num_kmers);

                std::size_t valid = 0;
                for_each_kmer_ascii_checked_chunked_simd<Simd, ChunkSize>(
                    seq, k, encoder, CheckedVectorShape<Simd>{ChunkSize, &valid}
                );
                EXPECT_EQ(valid, num_kmers);
            });
        }
    }
}

TEST(KmerExtractAsciiChunkedSimd, Scalar)
{
    check_chunked_simd<SimdScalar>();
    check_chunked_simd_vectors<SimdScalar>();
}

#if defined(FISK_HAS_SSE2)
TEST(KmerExtractAsciiChunkedSimd, Sse2)
{
    check_chunked_simd<SimdSSE2>();
    check_chunked_simd_vectors<SimdSSE2>();
}
#endif

#if defined(FISK_HAS_AVX2)
TEST(KmerExtractAsciiChunkedSimd, Avx2)
{
    check_chunked_simd<SimdAVX2>();
    check_chunked_simd_vectors<SimdAVX2>();
}
#endif

#if defined(FISK_HAS_AVX512)
TEST(KmerExtractAsciiChunkedSimd, Avx512)
{
    check_chunked_simd<SimdAVX512>();
    check_chunked_simd_vectors<SimdAVX512>();
}
#endif

#if defined(FISK_HAS_NEON)
TEST(KmerExtractAsciiChunkedSimd, Neon)
{
    check_chunked_simd<SimdNEON>();
    check_chunked_simd_vectors<SimdNEON>();
}
#endif

// The encoder is independent of the arch of the loop: a SIMD word encoder packs the chunks for
// the loop of another arch.
#if defined(FISK_HAS_SSE2) && defined(FISK_HAS_AVX2)
TEST(KmerExtractAsciiChunkedSimd, UncheckedEncoderOfOtherArch)
{
    constexpr Encoding E = Encoding::kACGT;
    constexpr Layout L = Layout::kMSB;
    WordEncoderDirect<E, L, SimdSSE2> const encoder;
    PackedSequence<E, L> scratch;
    for (auto const& seq : chunked_test_sequences()) {
        for (std::size_t const k : {std::size_t{1}, std::size_t{15}, std::size_t{31}}) {
            auto const expected = oracle_kmers_all_windows<E, L>(seq, k);
            check_kmer_callbacks<E, L>(expected, k, [&](auto const& callback) {
                unchecked<SimdAVX2>.template operator()<32>(seq, k, encoder, &scratch, callback);
            });
        }
    }
}
#endif
