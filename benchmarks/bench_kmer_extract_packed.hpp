#pragma once

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils.hpp"
#include "fisk/kmer_extract/packed.hpp"
#include "fisk/kmer_extract/packed_simd.hpp"
#include "fisk/seq_pack/seq_pack.hpp"
#include "fisk/core/types.hpp"
#include "microbench.hpp"

using namespace fisk;

// =================================================================================================
//     Sum Hashing
// =================================================================================================

// Benchmark sinks for the packed k-mer extractors in packed.hpp and packed_simd.hpp: sum every
// emitted k-mer into `hash`.

// ------------------------------------------------------------------------
//     Scalar
// ------------------------------------------------------------------------

// Named callback types keep direct and dispatcher rows on the same template instantiation. Using a
// separate lambda at each call site can make the compiler generate different kernel clones.
struct PackedScalarSum
{
    std::uint64_t value = 0;

    // Templated on the k-mer type rather than fixed to one convention, so that the same named
    // callback serves every PackedSequence instantiation benchmarked below.
    template <KmerType K>
    void operator()(K kmer) noexcept { value += kmer_value(kmer); }
};

template <Encoding E, Layout L>
inline std::uint64_t compute_kmer_hash_packed_rolling(
    PackedSequence<E, L> const& seq, std::size_t k
) {
    PackedScalarSum sum;
    for_each_kmer_packed_rolling(seq, k, sum);
    return sum.value;
}

template <Encoding E, Layout L>
inline std::uint64_t compute_kmer_hash_packed_aligned(
    PackedSequence<E, L> const& seq, std::size_t k
) {
    PackedScalarSum sum;
    for_each_kmer_packed_aligned(seq, k, sum);
    return sum.value;
}

// Same sum-hash sink as above, but built from the packed_simd.hpp vector-emitting extractors:
// accumulate into a persistent vector register across every call (unconditionally; zero-padded
// tail lanes add as 0, so `valid_count` needs no attention for a pure sum), then reduce to a scalar
// once at the end. Unsigned wraparound add is associative/commutative regardless of grouping, so
// this produces the exact same total as the scalar hash functions above, letting the sinks
// cross-validate every SIMD tier against every scalar variant for free.

// ------------------------------------------------------------------------
//     SSE2
// ------------------------------------------------------------------------

#if defined(FISK_HAS_SSE2)

struct PackedSse2Sum
{
    __m128i value = _mm_setzero_si128();

    void operator()(__m128i v, std::size_t) noexcept { value = _mm_add_epi64(value, v); }

    std::uint64_t result() const noexcept
    {
        alignas(16) std::uint64_t buf[2];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(buf), value);
        return buf[0] + buf[1];
    }
};

template <Encoding E, Layout L>
inline std::uint64_t compute_kmer_hash_packed_simd_narrow_sse2(
    PackedSequence<E, L> const& seq, std::size_t k
) {
    PackedSse2Sum sum;
    for_each_kmer_packed_simd_narrow_sse2_(seq, k, sum);
    return sum.result();
}

template <Encoding E, Layout L>
inline std::uint64_t compute_kmer_hash_packed_simd_wide_sse2(
    PackedSequence<E, L> const& seq, std::size_t k
) {
    PackedSse2Sum sum;
    for_each_kmer_packed_simd_wide_sse2_(seq, k, sum);
    return sum.result();
}

template <Encoding E, Layout L>
inline std::uint64_t compute_kmer_hash_packed_simd_sse2(
    PackedSequence<E, L> const& seq, std::size_t k
) {
    PackedSse2Sum sum;
    for_each_kmer_packed_simd_sse2(seq, k, sum);
    return sum.result();
}

#endif // FISK_HAS_SSE2

// ------------------------------------------------------------------------
//     AVX2
// ------------------------------------------------------------------------

#if defined(FISK_HAS_AVX2)

struct PackedAvx2Sum
{
    __m256i value = _mm256_setzero_si256();

    void operator()(__m256i v, std::size_t) noexcept { value = _mm256_add_epi64(value, v); }

    std::uint64_t result() const noexcept
    {
        alignas(32) std::uint64_t buf[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(buf), value);
        return buf[0] + buf[1] + buf[2] + buf[3];
    }
};

template <Encoding E, Layout L>
inline std::uint64_t compute_kmer_hash_packed_simd_narrow_avx2(
    PackedSequence<E, L> const& seq, std::size_t k
) {
    PackedAvx2Sum sum;
    for_each_kmer_packed_simd_narrow_avx2_(seq, k, sum);
    return sum.result();
}

template <Encoding E, Layout L>
inline std::uint64_t compute_kmer_hash_packed_simd_wide_avx2(
    PackedSequence<E, L> const& seq, std::size_t k
) {
    PackedAvx2Sum sum;
    for_each_kmer_packed_simd_wide_avx2_(seq, k, sum);
    return sum.result();
}

template <Encoding E, Layout L>
inline std::uint64_t compute_kmer_hash_packed_simd_avx2(
    PackedSequence<E, L> const& seq, std::size_t k
) {
    PackedAvx2Sum sum;
    for_each_kmer_packed_simd_avx2(seq, k, sum);
    return sum.result();
}

#endif // FISK_HAS_AVX2

// ------------------------------------------------------------------------
//     AVX-512
// ------------------------------------------------------------------------

#if defined(FISK_HAS_AVX512)

struct PackedAvx512Sum
{
    __m512i value = _mm512_setzero_si512();

    void operator()(__m512i v, std::size_t) noexcept { value = _mm512_add_epi64(value, v); }

    std::uint64_t result() const noexcept
    {
        alignas(64) std::uint64_t buf[8];
        _mm512_storeu_si512(buf, value);
        std::uint64_t sum = 0;
        for (auto x : buf) { sum += x; }
        return sum;
    }
};

template <Encoding E, Layout L>
inline std::uint64_t compute_kmer_hash_packed_simd_narrow_avx512(
    PackedSequence<E, L> const& seq, std::size_t k
) {
    PackedAvx512Sum sum;
    for_each_kmer_packed_simd_narrow_avx512_(seq, k, sum);
    return sum.result();
}

template <Encoding E, Layout L>
inline std::uint64_t compute_kmer_hash_packed_simd_wide_avx512(
    PackedSequence<E, L> const& seq, std::size_t k
) {
    PackedAvx512Sum sum;
    for_each_kmer_packed_simd_wide_avx512_(seq, k, sum);
    return sum.result();
}

template <Encoding E, Layout L>
inline std::uint64_t compute_kmer_hash_packed_simd_avx512(
    PackedSequence<E, L> const& seq, std::size_t k
) {
    PackedAvx512Sum sum;
    for_each_kmer_packed_simd_avx512(seq, k, sum);
    return sum.result();
}

#endif // FISK_HAS_AVX512

// ------------------------------------------------------------------------
//     NEON
// ------------------------------------------------------------------------

#if defined(FISK_HAS_NEON)

struct PackedNeonSum
{
    uint64x2_t value = vdupq_n_u64(0);

    void operator()(uint64x2_t v, std::size_t) noexcept { value = vaddq_u64(value, v); }

    std::uint64_t result() const noexcept
    {
        return vgetq_lane_u64(value, 0) + vgetq_lane_u64(value, 1);
    }
};

template <Encoding E, Layout L>
inline std::uint64_t compute_kmer_hash_packed_simd_narrow_neon(
    PackedSequence<E, L> const& seq, std::size_t k
) {
    PackedNeonSum sum;
    for_each_kmer_packed_simd_narrow_neon_(seq, k, sum);
    return sum.result();
}

template <Encoding E, Layout L>
inline std::uint64_t compute_kmer_hash_packed_simd_wide_neon(
    PackedSequence<E, L> const& seq, std::size_t k
) {
    PackedNeonSum sum;
    for_each_kmer_packed_simd_wide_neon_(seq, k, sum);
    return sum.result();
}

template <Encoding E, Layout L>
inline std::uint64_t compute_kmer_hash_packed_simd_neon(
    PackedSequence<E, L> const& seq, std::size_t k
) {
    PackedNeonSum sum;
    for_each_kmer_packed_simd_neon(seq, k, sum);
    return sum.result();
}

#endif // FISK_HAS_NEON

// =================================================================================================
//     Benchmark
// =================================================================================================

/**
 * @brief Benchmark k-mer extraction directly from a PackedSequence (kmer_extract/packed.hpp),
 * across both Layout conventions and both accumulator widths (narrow: single 64-bit register,
 * k<=29; wide: 128-bit-equivalent, k<=32).
 *
 * Packing is done once per sequence, outside the timed region: real usage would already have `seq`
 * packed ahead of time, so packing cost should not count against extraction cost here. See
 * bench_kmer_extract() (bench_kmer_extract.hpp) for the ASCII-input baseline to compare this
 * against, in the separate "kmer_extract" CSV suite.
 */
inline void bench_kmer_extract_packed(
    std::vector<std::string> const& sequences,
    std::size_t k_min,
    std::size_t k_max,
    std::ostream& csv_os
) {
    if (k_min < 1 || k_min > 32 || k_max < 1 || k_max > 32) {
        throw std::runtime_error("Invalid k outside of [1, 32]");
    }
    if (k_min > k_max) {
        throw std::runtime_error("Invalid k_min > k_max");
    }

    std::size_t const rounds = 8;
    std::size_t const repeats = 8;

    std::string const suite_title = "kmer_extract_packed";
    std::cout << "\n=== k-mer extract (packed) ===\n";
    std::cout << "rounds=" << rounds << ", repeats=" << repeats << "\n";

    write_csv_header(csv_os);

    // Every sequence here is packed with an ACGT word encoder; only the layout varies.
    using PackedMsb = PackedSequence<Encoding::kACGT, Layout::kMSB>;
    using PackedLsb = PackedSequence<Encoding::kACGT, Layout::kLSB>;

    std::vector<PackedMsb> packed_msb;
    std::vector<PackedLsb> packed_lsb;
    packed_msb.reserve(sequences.size());
    packed_lsb.reserve(sequences.size());
    for (auto const& seq : sequences) {
        packed_msb.push_back(pack_sequence(seq, WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>{}));
        packed_lsb.push_back(pack_sequence(seq, WordEncoderButterfly<Encoding::kACGT, Layout::kLSB>{}));
    }

    std::size_t const narrow_k_max = std::min<std::size_t>(k_max, 29);

    // -----------------------------------------------------------------------
    //     layout=msb, k in [k_min, min(k_max, 29)]: narrow and wide
    // -----------------------------------------------------------------------
    for (std::size_t k = k_min; k <= narrow_k_max; ++k) {
        if (stdout_is_terminal()) {
            std::cout << "\rlayout=msb k " << std::setw(2) << k << std::flush;
        }
        Microbench<PackedMsb> suite(suite_title);
        suite.rounds(rounds).repeats(repeats).units_fn(
            [k](PackedMsb const& seq) {
                return static_cast<double>(seq.length - k + 1);
            }
        );
        auto results = suite.run(
            packed_msb,
            #if defined(FISK_HAS_SSE2)
            bench("simd_narrow_sse2", [&](PackedMsb const& seq) {
                return compute_kmer_hash_packed_simd_narrow_sse2(seq, k);
            }),
            bench("simd_wide_sse2", [&](PackedMsb const& seq) {
                return compute_kmer_hash_packed_simd_wide_sse2(seq, k);
            }),
            bench("simd_sse2", [&](PackedMsb const& seq) {
                return compute_kmer_hash_packed_simd_sse2(seq, k);
            }),
            #endif // FISK_HAS_SSE2
            #if defined(FISK_HAS_AVX2)
            bench("simd_narrow_avx2", [&](PackedMsb const& seq) {
                return compute_kmer_hash_packed_simd_narrow_avx2(seq, k);
            }),
            bench("simd_wide_avx2", [&](PackedMsb const& seq) {
                return compute_kmer_hash_packed_simd_wide_avx2(seq, k);
            }),
            bench("simd_avx2", [&](PackedMsb const& seq) {
                return compute_kmer_hash_packed_simd_avx2(seq, k);
            }),
            #endif // FISK_HAS_AVX2
            #if defined(FISK_HAS_AVX512)
            bench("simd_narrow_avx512", [&](PackedMsb const& seq) {
                return compute_kmer_hash_packed_simd_narrow_avx512(seq, k);
            }),
            bench("simd_wide_avx512", [&](PackedMsb const& seq) {
                return compute_kmer_hash_packed_simd_wide_avx512(seq, k);
            }),
            bench("simd_avx512", [&](PackedMsb const& seq) {
                return compute_kmer_hash_packed_simd_avx512(seq, k);
            }),
            #endif // FISK_HAS_AVX512
            #if defined(FISK_HAS_NEON)
            bench("simd_narrow_neon", [&](PackedMsb const& seq) {
                return compute_kmer_hash_packed_simd_narrow_neon(seq, k);
            }),
            bench("simd_wide_neon", [&](PackedMsb const& seq) {
                return compute_kmer_hash_packed_simd_wide_neon(seq, k);
            }),
            bench("simd_neon", [&](PackedMsb const& seq) {
                return compute_kmer_hash_packed_simd_neon(seq, k);
            }),
            #endif // FISK_HAS_NEON
            bench("aligned", [&](PackedMsb const& seq) {
                return compute_kmer_hash_packed_aligned(seq, k);
            }),
            bench("rolling", [&](PackedMsb const& seq) {
                return compute_kmer_hash_packed_rolling(seq, k);
            })
        );
        write_csv_rows(csv_os, suite_title, "layout=msb;k=" + std::to_string(k), results);
    }

    // -----------------------------------------------------------------------
    //     layout=msb, k in [30, k_max]: wide only
    // -----------------------------------------------------------------------
    for (std::size_t k = std::max<std::size_t>(k_min, 30); k <= k_max; ++k) {
        if (stdout_is_terminal()) {
            std::cout << "\rlayout=msb k " << std::setw(2) << k << std::flush;
        }
        Microbench<PackedMsb> suite(suite_title);
        suite.rounds(rounds).repeats(repeats).units_fn(
            [k](PackedMsb const& seq) {
                return static_cast<double>(seq.length - k + 1);
            }
        );
        auto results = suite.run(
            packed_msb,
            #if defined(FISK_HAS_SSE2)
            bench("simd_wide_sse2", [&](PackedMsb const& seq) {
                return compute_kmer_hash_packed_simd_wide_sse2(seq, k);
            }),
            bench("simd_sse2", [&](PackedMsb const& seq) {
                return compute_kmer_hash_packed_simd_sse2(seq, k);
            }),
            #endif // FISK_HAS_SSE2
            #if defined(FISK_HAS_AVX2)
            bench("simd_wide_avx2", [&](PackedMsb const& seq) {
                return compute_kmer_hash_packed_simd_wide_avx2(seq, k);
            }),
            bench("simd_avx2", [&](PackedMsb const& seq) {
                return compute_kmer_hash_packed_simd_avx2(seq, k);
            }),
            #endif // FISK_HAS_AVX2
            #if defined(FISK_HAS_AVX512)
            bench("simd_wide_avx512", [&](PackedMsb const& seq) {
                return compute_kmer_hash_packed_simd_wide_avx512(seq, k);
            }),
            bench("simd_avx512", [&](PackedMsb const& seq) {
                return compute_kmer_hash_packed_simd_avx512(seq, k);
            }),
            #endif // FISK_HAS_AVX512
            #if defined(FISK_HAS_NEON)
            bench("simd_wide_neon", [&](PackedMsb const& seq) {
                return compute_kmer_hash_packed_simd_wide_neon(seq, k);
            }),
            bench("simd_neon", [&](PackedMsb const& seq) {
                return compute_kmer_hash_packed_simd_neon(seq, k);
            }),
            #endif // FISK_HAS_NEON
            bench("aligned", [&](PackedMsb const& seq) {
                return compute_kmer_hash_packed_aligned(seq, k);
            }),
            bench("rolling", [&](PackedMsb const& seq) {
                return compute_kmer_hash_packed_rolling(seq, k);
            })
        );
        write_csv_rows(csv_os, suite_title, "layout=msb;k=" + std::to_string(k), results);
    }

    // -----------------------------------------------------------------------
    //     layout=lsb, k in [k_min, min(k_max, 29)]: narrow and wide
    // -----------------------------------------------------------------------
    for (std::size_t k = k_min; k <= narrow_k_max; ++k) {
        if (stdout_is_terminal()) {
            std::cout << "\rlayout=lsb k=" << std::setw(2) << k << std::flush;
        }
        Microbench<PackedLsb> suite(suite_title);
        suite.rounds(rounds).repeats(repeats).units_fn(
            [k](PackedLsb const& seq) {
                return static_cast<double>(seq.length - k + 1);
            }
        );
        auto results = suite.run(
            packed_lsb,
            #if defined(FISK_HAS_SSE2)
            bench("simd_narrow_sse2", [&](PackedLsb const& seq) {
                return compute_kmer_hash_packed_simd_narrow_sse2(seq, k);
            }),
            bench("simd_wide_sse2", [&](PackedLsb const& seq) {
                return compute_kmer_hash_packed_simd_wide_sse2(seq, k);
            }),
            bench("simd_sse2", [&](PackedLsb const& seq) {
                return compute_kmer_hash_packed_simd_sse2(seq, k);
            }),
            #endif // FISK_HAS_SSE2
            #if defined(FISK_HAS_AVX2)
            bench("simd_narrow_avx2", [&](PackedLsb const& seq) {
                return compute_kmer_hash_packed_simd_narrow_avx2(seq, k);
            }),
            bench("simd_wide_avx2", [&](PackedLsb const& seq) {
                return compute_kmer_hash_packed_simd_wide_avx2(seq, k);
            }),
            bench("simd_avx2", [&](PackedLsb const& seq) {
                return compute_kmer_hash_packed_simd_avx2(seq, k);
            }),
            #endif // FISK_HAS_AVX2
            #if defined(FISK_HAS_AVX512)
            bench("simd_narrow_avx512", [&](PackedLsb const& seq) {
                return compute_kmer_hash_packed_simd_narrow_avx512(seq, k);
            }),
            bench("simd_wide_avx512", [&](PackedLsb const& seq) {
                return compute_kmer_hash_packed_simd_wide_avx512(seq, k);
            }),
            bench("simd_avx512", [&](PackedLsb const& seq) {
                return compute_kmer_hash_packed_simd_avx512(seq, k);
            }),
            #endif // FISK_HAS_AVX512
            #if defined(FISK_HAS_NEON)
            bench("simd_narrow_neon", [&](PackedLsb const& seq) {
                return compute_kmer_hash_packed_simd_narrow_neon(seq, k);
            }),
            bench("simd_wide_neon", [&](PackedLsb const& seq) {
                return compute_kmer_hash_packed_simd_wide_neon(seq, k);
            }),
            bench("simd_neon", [&](PackedLsb const& seq) {
                return compute_kmer_hash_packed_simd_neon(seq, k);
            }),
            #endif // FISK_HAS_NEON
            bench("aligned", [&](PackedLsb const& seq) {
                return compute_kmer_hash_packed_aligned(seq, k);
            }),
            bench("rolling", [&](PackedLsb const& seq) {
                return compute_kmer_hash_packed_rolling(seq, k);
            })
        );
        write_csv_rows(csv_os, suite_title, "layout=lsb;k=" + std::to_string(k), results);
    }

    // -----------------------------------------------------------------------
    //     layout=lsb, k in [30, k_max]: wide only
    // -----------------------------------------------------------------------
    for (std::size_t k = std::max<std::size_t>(k_min, 30); k <= k_max; ++k) {
        if (stdout_is_terminal()) {
            std::cout << "\rlayout=lsb k=" << std::setw(2) << k << std::flush;
        }
        Microbench<PackedLsb> suite(suite_title);
        suite.rounds(rounds).repeats(repeats).units_fn(
            [k](PackedLsb const& seq) {
                return static_cast<double>(seq.length - k + 1);
            }
        );
        auto results = suite.run(
            packed_lsb,
            #if defined(FISK_HAS_SSE2)
            bench("simd_wide_sse2", [&](PackedLsb const& seq) {
                return compute_kmer_hash_packed_simd_wide_sse2(seq, k);
            }),
            bench("simd_sse2", [&](PackedLsb const& seq) {
                return compute_kmer_hash_packed_simd_sse2(seq, k);
            }),
            #endif // FISK_HAS_SSE2
            #if defined(FISK_HAS_AVX2)
            bench("simd_wide_avx2", [&](PackedLsb const& seq) {
                return compute_kmer_hash_packed_simd_wide_avx2(seq, k);
            }),
            bench("simd_avx2", [&](PackedLsb const& seq) {
                return compute_kmer_hash_packed_simd_avx2(seq, k);
            }),
            #endif // FISK_HAS_AVX2
            #if defined(FISK_HAS_AVX512)
            bench("simd_wide_avx512", [&](PackedLsb const& seq) {
                return compute_kmer_hash_packed_simd_wide_avx512(seq, k);
            }),
            bench("simd_avx512", [&](PackedLsb const& seq) {
                return compute_kmer_hash_packed_simd_avx512(seq, k);
            }),
            #endif // FISK_HAS_AVX512
            #if defined(FISK_HAS_NEON)
            bench("simd_wide_neon", [&](PackedLsb const& seq) {
                return compute_kmer_hash_packed_simd_wide_neon(seq, k);
            }),
            bench("simd_neon", [&](PackedLsb const& seq) {
                return compute_kmer_hash_packed_simd_neon(seq, k);
            }),
            #endif // FISK_HAS_NEON
            bench("aligned", [&](PackedLsb const& seq) {
                return compute_kmer_hash_packed_aligned(seq, k);
            }),
            bench("rolling", [&](PackedLsb const& seq) {
                return compute_kmer_hash_packed_rolling(seq, k);
            })
        );
        write_csv_rows(csv_os, suite_title, "layout=lsb;k=" + std::to_string(k), results);
    }

    if (stdout_is_terminal()) {
        std::cout << "\n";
    }
}

inline void bench_kmer_extract_packed(
    std::vector<std::string> const& sequences,
    std::ostream& csv_os
) {
    bench_kmer_extract_packed(sequences, 1, 32, csv_os);
}
