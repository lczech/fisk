#pragma once

// SIMD vectors of raw k-mer words for the vector casts, kmer_vector_cast() and the vector overloads
// of spaced_kmer_cast(), and the check of the latter. Each vector is built with its ISA's own set
// intrinsic rather than via memcpy, since it is precisely the lane order of those intrinsics that
// the casts and the functions on k-mer vectors must agree with, and memcpy is what they use
// internally.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/core/cast.hpp"
#include "fisk/core/intrinsics.hpp"
#include "fisk/core/kmer.hpp"
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
//     Vector Cast Check
// =================================================================================================

// The vector overloads of spaced_kmer_cast() and spaced_kmer_cast_unchecked() must hand back
// exactly the lanes they were given, in lane order and tagged with K's conventions, so that
// consumers of the vector-emitting spaced extractors can return to the typed world. Lane `i` of
// `vec` holds seqs[i] encoded under the conventions of K, see for_each_lane_vector().
template <typename K, typename Vec>
inline void check_vector_cast(Vec vec, std::vector<std::string> const& seqs, std::size_t width)
{
    constexpr std::size_t lanes = sizeof(Vec) / sizeof(std::uint64_t);
    ASSERT_EQ(seqs.size(), lanes);

    auto const cast = [&](auto... args) {
        return fisk::spaced_kmer_cast<K::encoding, K::layout>(vec, args...);
    };
    auto const cast_unchecked = [&](auto... args) {
        return fisk::spaced_kmer_cast_unchecked<K::encoding, K::layout>(vec, args...);
    };

    // The full-vector overloads write every lane, checked and unchecked alike.
    K full[lanes] = {};
    K full_unchecked[lanes] = {};
    EXPECT_EQ(cast(width, full), lanes);
    EXPECT_EQ(cast_unchecked(full_unchecked), lanes);
    for (std::size_t i = 0; i < lanes; ++i) {
        EXPECT_EQ(full[i], oracle_kmer<K>(seqs[i]));
        EXPECT_EQ(full_unchecked[i], full[i]);
        EXPECT_EQ(kmer_decode(full[i], width), seqs[i]);
    }

    // The counted overloads write only the lanes they are told are valid, as the zero-padded final
    // vector of an extraction run requires.
    K partial[lanes] = {};
    K partial_unchecked[lanes] = {};
    EXPECT_EQ(cast(std::size_t{1}, width, partial), std::size_t{1});
    EXPECT_EQ(cast_unchecked(std::size_t{1}, partial_unchecked), std::size_t{1});
    EXPECT_EQ(partial[0], oracle_kmer<K>(seqs[0]));
    EXPECT_EQ(partial_unchecked[0], partial[0]);
    if constexpr (lanes > 1) {
        EXPECT_EQ(partial[1], K{0});
        EXPECT_EQ(partial_unchecked[1], K{0});
    }
}
