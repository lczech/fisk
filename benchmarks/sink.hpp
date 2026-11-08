#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/core/cast.hpp"
#include "fisk/core/types.hpp"
#include "fisk/util/intrinsics.hpp"

// Selects which of three strategies every run_var_*() function's per-element accumulation
// compiles against. One build, one strategy: change the value below, or override the
// FISK_BENCHMARK_SINK_MODE CMake cache variable (see CMakeLists.txt) and rebuild, to compare Sum
// vs Barrier vs Write.
//
// This macro being defined at all -- regardless of which of the three values it holds -- is also
// what a couple of headers under include/ check to enable benchmark-only correctness barriers
// (see e.g. kmer_extract/internal.hpp) that would otherwise cost regular library users a small,
// unnecessary register-pressure tax. CMakeLists.txt defines it for every benchmark TU from the
// start of preprocessing (not relying on include order), and never defines it for fisk::fisk
// itself or its consumers.
//
//   Sum     -- the default: consume(v) accumulates into a running sum, basically `hash += v`.
//              Free to auto-vectorize; measures the natural "best" case speed of the technique.
//   Barrier -- forces each per-element result through do_not_optimize() individually, instead of
//              letting them combine freely. Blocks auto-vectorization across elements, showing
//              the cost of a technique whose results cannot be batched; the "pessimistic" case.
//   Write   -- writes each result into a small buffer instead of accumulating it, modeling a
//              technique that must actually store its output rather than fold it into one
//              running value. This might still vectorize, but reflects the "realstic" case.
//
// The run_var_*() functions mark their callbacks with FISK_ALWAYS_INLINE_LAMBDA, so that each
// consume() is part of the extraction loop, as with a small callback that the compiler inlines,
// instead of a call per k-mer that depends on the inlining heuristics of the compiler.
//
// The buffer that Write touches (and that Sum/Barrier ignore) comes from make_sink_buffer(): the
// same small size for all suites, independent of the input, so that it stays in the L2 cache
// along with the input, and the writes wrap around in it. The buffer is owned by the caller and
// outlives each call, so the compiler has to keep the writes, and the caller's per-round
// clobber_memory() (see e.g. bit_extract_weights/var_*.cpp) makes every round's writes genuine,
// so consume() itself needs no barrier of its own.
//
// Overloaded for the SIMD register types too (guarded the same way as do_not_optimize()'s own
// overloads), for extractors whose whole point is to hand the caller a register of several
// k-mers at once. Benchmarking them by extracting individual lanes would measure work the
// technique was never meant to do. Sum accumulates whole registers via a per-width vector
// accumulator, reduced to one scalar only in finalize(); Barrier barriers the whole register, not
// its lanes; Write stores the whole register per call, advancing by its lane count. The vector
// variants pass their registers through consume_valid_lanes(), see below, which sets the invalid
// lanes to zero, so that all variants of a suite sum the same values.
#define FISK_BENCHMARK_SINK_SUM     0
#define FISK_BENCHMARK_SINK_BARRIER 1
#define FISK_BENCHMARK_SINK_WRITE   2

#ifndef FISK_BENCHMARK_SINK_MODE
#define FISK_BENCHMARK_SINK_MODE FISK_BENCHMARK_SINK_SUM
#endif

// =================================================================================================
//     FISK_BENCHMARK_SINK_SUM
// =================================================================================================

#if FISK_BENCHMARK_SINK_MODE == FISK_BENCHMARK_SINK_SUM

inline constexpr char const* kSinkName = "sum";

// The vector accumulators below follow the scalar one, so that MSVC pads the struct to their
// alignment, and warns about that (C4324). The padding is intended.
#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable: 4324)
#endif
struct Sink
{
    std::uint64_t acc = 0;

    void consume(std::uint64_t v) noexcept
    {
        acc += v;
    }

    #if defined(FISK_HAS_SSE2)

        __m128i acc_128 = _mm_setzero_si128();
        void consume(__m128i v) noexcept
        {
            acc_128 = _mm_add_epi64(acc_128, v);
        }

    #endif
    #if defined(FISK_HAS_AVX2)

        __m256i acc_256 = _mm256_setzero_si256();
        void consume(__m256i v) noexcept
        {
            acc_256 = _mm256_add_epi64(acc_256, v);
        }

    #endif
    #if defined(FISK_HAS_AVX512)

        __m512i acc_512 = _mm512_setzero_si512();
        void consume(__m512i v) noexcept
        {
            acc_512 = _mm512_add_epi64(acc_512, v);
        }

    #endif
    #if defined(FISK_HAS_NEON)

        uint64x2_t acc_neon = vdupq_n_u64(0);
        void consume(uint64x2_t v) noexcept
        {
            acc_neon = vaddq_u64(acc_neon, v);
        }

    #endif

    std::uint64_t finalize() const noexcept
    {
        std::uint64_t s = acc;
        #if defined(FISK_HAS_SSE2)
        {
            alignas(16) std::uint64_t lanes[2];
            _mm_store_si128(reinterpret_cast<__m128i*>(lanes), acc_128);
            s += lanes[0] + lanes[1];
        }
        #endif
        #if defined(FISK_HAS_AVX2)
        {
            alignas(32) std::uint64_t lanes[4];
            _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc_256);
            s += lanes[0] + lanes[1] + lanes[2] + lanes[3];
        }
        #endif
        #if defined(FISK_HAS_AVX512)
        {
            alignas(64) std::uint64_t lanes[8];
            _mm512_store_si512(reinterpret_cast<void*>(lanes), acc_512);
            for (auto v : lanes) s += v;
        }
        #endif
        #if defined(FISK_HAS_NEON)
        {
            alignas(16) std::uint64_t lanes[2];
            vst1q_u64(lanes, acc_neon);
            s += lanes[0] + lanes[1];
        }
        #endif
        return s;
    }
};
#if defined(_MSC_VER)
    #pragma warning(pop)
#endif

inline Sink make_sink(std::vector<std::uint64_t>&)
{
    return Sink{};
}

// =================================================================================================
//     FISK_BENCHMARK_SINK_BARRIER
// =================================================================================================

#elif FISK_BENCHMARK_SINK_MODE == FISK_BENCHMARK_SINK_BARRIER

inline constexpr char const* kSinkName = "barrier";

struct Sink
{
    void consume(std::uint64_t v) const noexcept
    {
        fisk::do_not_optimize(v);
    }

    #if defined(FISK_HAS_SSE2)

        void consume(__m128i v) const noexcept
        {
            fisk::do_not_optimize(v);
        }

    #endif
    #if defined(FISK_HAS_AVX2)

        void consume(__m256i v) const noexcept
        {
            fisk::do_not_optimize(v);
        }

    #endif
    #if defined(FISK_HAS_AVX512)

        void consume(__m512i v) const noexcept
        {
            fisk::do_not_optimize(v);
        }

    #endif
    #if defined(FISK_HAS_NEON)

        void consume(uint64x2_t v) const noexcept
        {
            fisk::do_not_optimize(v);
        }

    #endif

    std::uint64_t finalize() const noexcept
    {
        // legitimately no correctness value
        return 0;
    }
};

inline Sink make_sink(std::vector<std::uint64_t>&)
{
    return Sink{};
}

// =================================================================================================
//     FISK_BENCHMARK_SINK_WRITE
// =================================================================================================

#elif FISK_BENCHMARK_SINK_MODE == FISK_BENCHMARK_SINK_WRITE

inline constexpr char const* kSinkName = "write";

struct Sink
{
    std::vector<std::uint64_t>& buf;
    std::size_t mask;

    // The index of the single values, and one per register width, see consume() of the registers.
    std::size_t idx = 0;
    std::size_t idx_128 = 0;
    std::size_t idx_256 = 0;
    std::size_t idx_512 = 0;

    void consume(std::uint64_t v) noexcept
    {
        buf[idx++ & mask] = v;
    }

    // Store the whole register at once and advance the index of its width by its lane count. Each
    // width has an index of its own, so that it stays a multiple of the lane count also in a sink
    // that takes single values and registers, such as the one of tune(), which calls the scalar
    // and the SIMD algorithms with one callback (see extractor_scaling/variant.hpp). With buf's
    // size a power of two, and so a multiple of every lane count below (2/4/8), a store then
    // always has a full register's worth of room ahead of it before the next wrap.
    #if defined(FISK_HAS_SSE2)

        void consume(__m128i v) noexcept
        {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&buf[idx_128 & mask]), v);
            idx_128 += 2;
        }

    #endif
    #if defined(FISK_HAS_AVX2)

        void consume(__m256i v) noexcept
        {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&buf[idx_256 & mask]), v);
            idx_256 += 4;
        }

    #endif
    #if defined(FISK_HAS_AVX512)

        void consume(__m512i v) noexcept
        {
            _mm512_storeu_si512(reinterpret_cast<void*>(&buf[idx_512 & mask]), v);
            idx_512 += 8;
        }

    #endif
    #if defined(FISK_HAS_NEON)

        void consume(uint64x2_t v) noexcept
        {
            vst1q_u64(&buf[idx_128 & mask], v);
            idx_128 += 2;
        }

    #endif

    // No correctness value, as for Barrier: the buffer only keeps the last writes, which differ
    // between the scalar variants and the vector ones that also write their invalid lanes. Reading
    // the buffer here would also add a pass over it to the timed call.
    std::uint64_t finalize() const noexcept
    {
        return 0;
    }
};

inline Sink make_sink(std::vector<std::uint64_t>& buf)
{
    return Sink{buf, buf.size() - 1, 0, 0, 0, 0};
}

// =================================================================================================
//     Unknown sink
// =================================================================================================

#else
#error "Unknown FISK_BENCHMARK_SINK_MODE"
#endif

// =================================================================================================
//     Sink buffer
// =================================================================================================

/**
 * @brief The buffer for make_sink(), the same for all suites: 64 KiB, which stays in the L2 cache
 * along with the input, so that Write measures the stores rather than memory throughput.
 *
 * Its size is a power of two, so that the writes wrap around with a mask, and a multiple of every
 * vector lane count, see Sink::consume().
 */
inline std::vector<std::uint64_t> make_sink_buffer()
{
    return std::vector<std::uint64_t>(std::size_t{1} << 13, 0);
}

// =================================================================================================
//     Valid lanes
// =================================================================================================

/**
 * @brief The vector `v` of the arch `Simd`, with the lanes outside of `mask` set to zero.
 */
template <fisk::SimdArch Simd>
FISK_ALWAYS_INLINE
inline typename Simd::simd_vector zero_invalid_lanes(
    typename Simd::simd_vector v, typename Simd::lane_mask mask
) noexcept {
    // A lane mask as a vector, or the mask register of AVX512, see SimdArch.
    if constexpr (sizeof(typename Simd::lane_mask) == sizeof(typename Simd::simd_vector)) {
        return Simd::bitwise_and(v, mask);
    } else {
        #if defined(FISK_HAS_AVX512)
        return _mm512_maskz_mov_epi64(mask, v);
        #endif
    }
}

/**
 * @brief Consume the valid lanes of a vector `v` of the arch `Simd`, as handed out by the
 * extraction functions without checks, with their count of valid lanes.
 *
 * Only the lanes below the count go into the sink, the others as zero, so that the sum is that of
 * the single k-mers. All lanes are valid except in the last vector, so the whole vector goes in
 * as it is unless the count is below all lanes, where a caller of the functions without checks
 * does not need to select the lanes.
 */
template <fisk::SimdArch Simd, typename Sink>
FISK_ALWAYS_INLINE
inline void consume_valid_lanes(
    Sink& sink, typename Simd::simd_vector v, std::size_t valid_count
) noexcept {
    if (valid_count == Simd::lanes) {
        sink.consume(v);
    } else {
        auto const bits = (std::uint64_t{1} << valid_count) - 1u;
        auto const mask = fisk::SimdVectorValidLanes<Simd>{bits}.lane_mask();
        sink.consume(zero_invalid_lanes<Simd>(v, mask));
    }
}

/**
 * @brief Consume the valid lanes of a vector `v` of the arch `Simd`, as handed out by the
 * extraction functions with checks, with their valid lanes.
 *
 * Only the valid lanes go into the sink, the others as zero, so that the sum is that of the single
 * valid k-mers. The lanes are selected in every vector, as which of them are valid depends on the
 * input.
 */
template <fisk::SimdArch Simd, typename Sink>
FISK_ALWAYS_INLINE
inline void consume_valid_lanes(
    Sink& sink, typename Simd::simd_vector v, fisk::SimdVectorValidLanes<Simd> valid_lanes
) noexcept {
    sink.consume(zero_invalid_lanes<Simd>(v, valid_lanes.lane_mask()));
}

/**
 * @brief Consume the valid lanes of a vector of k-mers or spaced k-mers, with their count of valid
 * lanes or their valid lanes, as above.
 */
template <typename Sink, fisk::KmerVectorType V, typename Valid>
FISK_ALWAYS_INLINE
inline void consume_valid_lanes(Sink& sink, V kmers, Valid valid) noexcept
{
    consume_valid_lanes<typename V::simd>(sink, fisk::kmer_value(kmers), valid);
}
