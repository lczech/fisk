#pragma once

// SIMD vectors of raw k-mer words for the vectors of k-mers and spaced k-mers, KmerVector and
// SpacedKmerVector, and the check of their casts and of the functions that take their lanes out.
// Each vector is built with its ISA's own set intrinsic rather than via memcpy, since it is
// precisely the lane order of those intrinsics that the casts and the functions on k-mer vectors
// must agree with, and memcpy is what they use internally.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "fisk/core/cast.hpp"
#include "fisk/core/intrinsics.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/simd_avx2.hpp"
#include "fisk/core/simd_avx512.hpp"
#include "fisk/core/simd_neon.hpp"
#include "fisk/core/simd_sse2.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "testing.hpp"

// =================================================================================================
//     Lane Vectors
// =================================================================================================

// Raw word of lane `i` under E/L, encoded by the oracle.
template <fisk::Encoding E, fisk::Layout L>
inline std::uint64_t lane_word(std::vector<std::string> const& seqs, std::size_t i)
{
    return oracle_encode<E, L>(seqs[i]).value();
}

// Call `f(simd, vec, seqs)` once per vector ISA compiled in, with `simd` an object of its SIMD
// arch, such as SimdAVX2. `seqs` holds one random sequence of length `width` per lane, and lane `i`
// of `vec` holds seqs[i] encoded under E/L.
template <fisk::Encoding E, fisk::Layout L, typename F>
inline void for_each_lane_vector(
    [[maybe_unused]] std::size_t width, [[maybe_unused]] std::uint64_t seed, [[maybe_unused]] F&& f
) {
    #if defined(FISK_HAS_SSE2)
    {
        auto const seqs = random_seqs(width, 2, seed);
        f(fisk::SimdSSE2{}, _mm_set_epi64x(
            static_cast<long long>(lane_word<E, L>(seqs, 1)),
            static_cast<long long>(lane_word<E, L>(seqs, 0))
        ), seqs);
    }
    #endif

    #if defined(FISK_HAS_AVX2)
    {
        auto const seqs = random_seqs(width, 4, seed + 1);
        f(fisk::SimdAVX2{}, _mm256_set_epi64x(
            static_cast<long long>(lane_word<E, L>(seqs, 3)),
            static_cast<long long>(lane_word<E, L>(seqs, 2)),
            static_cast<long long>(lane_word<E, L>(seqs, 1)),
            static_cast<long long>(lane_word<E, L>(seqs, 0))
        ), seqs);
    }
    #endif

    #if defined(FISK_HAS_AVX512)
    {
        auto const seqs = random_seqs(width, 8, seed + 2);
        f(fisk::SimdAVX512{}, _mm512_set_epi64(
            static_cast<long long>(lane_word<E, L>(seqs, 7)),
            static_cast<long long>(lane_word<E, L>(seqs, 6)),
            static_cast<long long>(lane_word<E, L>(seqs, 5)),
            static_cast<long long>(lane_word<E, L>(seqs, 4)),
            static_cast<long long>(lane_word<E, L>(seqs, 3)),
            static_cast<long long>(lane_word<E, L>(seqs, 2)),
            static_cast<long long>(lane_word<E, L>(seqs, 1)),
            static_cast<long long>(lane_word<E, L>(seqs, 0))
        ), seqs);
    }
    #endif

    #if defined(FISK_HAS_NEON)
    {
        auto const seqs = random_seqs(width, 2, seed + 3);
        std::uint64_t const lanes[2] = {lane_word<E, L>(seqs, 0), lane_word<E, L>(seqs, 1)};
        f(fisk::SimdNEON{}, vld1q_u64(lanes), seqs);
    }
    #endif
}

// =================================================================================================
//     K-mer Vector Check
// =================================================================================================

// The checked and unchecked vector casts of the k-mer type K: kmer_vector_cast() for a Kmer, and
// spaced_kmer_vector_cast() for a SpacedKmer.
template <typename K, typename Simd, typename Vec>
inline auto kmer_vector_cast_of(Vec vec, std::size_t width)
{
    if constexpr (fisk::SpacedKmerType<K>) {
        return fisk::spaced_kmer_vector_cast<K::encoding, K::layout, Simd>(vec, width);
    } else {
        return fisk::kmer_vector_cast<K::encoding, K::layout, Simd>(vec, width);
    }
}
template <typename K, typename Simd, typename Vec>
inline auto kmer_vector_cast_unchecked_of(Vec vec)
{
    if constexpr (fisk::SpacedKmerType<K>) {
        return fisk::spaced_kmer_vector_cast_unchecked<K::encoding, K::layout, Simd>(vec);
    } else {
        return fisk::kmer_vector_cast_unchecked<K::encoding, K::layout, Simd>(vec);
    }
}

// A vector of k-mers or spaced k-mers K holds the lanes it is cast from. for_each_valid_lane()
// hands out exactly its valid k-mers, in lane order and with their positions, for each count of
// valid lanes and each mask of valid lanes, in both callback forms, and store_valid_lanes() stores
// the same k-mers. Run once per vector ISA compiled in, see for_each_lane_vector().
template <typename K>
inline void check_kmer_vector(std::uint64_t seed)
{
    constexpr fisk::Encoding E = K::encoding;
    constexpr fisk::Layout L = K::layout;
    std::size_t const width = 10;
    std::size_t const pos = 100;
    for_each_lane_vector<E, L>(width, seed, [&]<typename Simd>(Simd, auto vec, auto const& seqs) {
        constexpr std::size_t lanes = Simd::lanes;
        auto const kmers = kmer_vector_cast_of<K, Simd>(vec, width);
        auto const kmers_unchecked = kmer_vector_cast_unchecked_of<K, Simd>(vec);
        static_assert(std::is_same_v<typename decltype(kmers)::kmer_type, K>);
        static_assert(std::is_same_v<decltype(kmers), decltype(kmers_unchecked)>);
        auto const value = fisk::kmer_value(kmers_unchecked);
        EXPECT_EQ(std::memcmp(&kmers.value, &vec, sizeof(vec)), 0);
        EXPECT_EQ(std::memcmp(&value, &vec, sizeof(vec)), 0);

        // The k-mers that each way hands out, for `valid`, which has the lanes of `bits`.
        auto const check = [&](auto const& valid, std::uint64_t bits) {
            std::vector<std::size_t> positions;
            std::vector<K> with_pos;
            std::vector<K> without_pos;
            fisk::for_each_valid_lane(pos, kmers, valid, [&](std::size_t p, K kmer) {
                positions.push_back(p);
                with_pos.push_back(kmer);
            });
            fisk::for_each_valid_lane(pos, kmers, valid, [&](K kmer) {
                without_pos.push_back(kmer);
            });
            K stored[lanes] = {};
            std::size_t const count = fisk::store_valid_lanes(kmers, valid, stored);

            std::size_t i = 0;
            for (std::size_t j = 0; j < lanes; ++j) {
                if (((bits >> j) & 1) == 0) {
                    continue;
                }
                ASSERT_TRUE(i < positions.size());
                EXPECT_EQ(positions[i], pos + j);
                EXPECT_EQ(with_pos[i], oracle_kmer<K>(seqs[j]));
                EXPECT_EQ(fisk::kmer_decode(with_pos[i], width), seqs[j]);
                EXPECT_EQ(without_pos[i], with_pos[i]);
                EXPECT_EQ(stored[i], with_pos[i]);
                ++i;
            }
            EXPECT_EQ(positions.size(), i);
            EXPECT_EQ(without_pos.size(), i);
            EXPECT_EQ(count, i);
        };
        for (std::size_t count = 0; count <= lanes; ++count) {
            check(count, (std::uint64_t{1} << count) - 1);
        }
        for (std::uint64_t bits = 0; bits < (std::uint64_t{1} << lanes); ++bits) {
            check(fisk::SimdVectorValidLanes<Simd>{bits}, bits);
        }
    });
}
