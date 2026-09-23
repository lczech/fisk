#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/core/intrinsics.hpp"

// Selects which of three strategies every run_var_*() function's per-element accumulation
// compiles against. One build, one strategy: change the value below, or override the
// FISK_BENCHMARK_SINK_MODE CMake cache variable (see CMakeLists.txt) and rebuild, to compare Sum
// vs Barrier vs Write.
//
// This macro being defined at all -- regardless of which of the three values it holds -- is also
// what a couple of headers under include/ check to enable benchmark-only correctness barriers
// (see e.g. kmer_extract/packed.hpp) that would otherwise cost regular library users a small,
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
// The buffer that Write touches (and that Sum/Barrier ignore) is sized to exactly match each
// call's inner batch, so one pass through the batch fills it exactly once with no wraparound; the
// caller's existing per-round clobber_memory() (see e.g. bit_extract_weights/var_*.cpp) already
// makes every subsequent round's writes genuine, so consume() itself needs no barrier of its own.
//
// Overloaded for the SIMD register types too (guarded the same way as do_not_optimize()'s own
// overloads), for extractors whose whole point is to hand the caller a register of several
// k-mers at once. Benchmarking them by extracting individual lanes would measure work the
// technique was never meant to do. Sum accumulates whole registers via a per-width vector
// accumulator, reduced to one scalar only in finalize(); Barrier barriers the whole register, not
// its lanes; Write stores the whole register per call, advancing by its lane count.
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
    std::size_t idx = 0;

    void consume(std::uint64_t v) noexcept
    {
        buf[idx++ & mask] = v;
    }

    // Store the whole register at once and advance by its lane count. Safe from a wrap splitting
    // one store across the buffer's end: buf's size is a power of two chosen to also be a
    // multiple of every lane count below (2/4/8), so idx & mask always lands on a lane-count-
    // aligned offset with a full register's worth of room ahead of it before the next wrap.
    #if defined(FISK_HAS_SSE2)

        void consume(__m128i v) noexcept
        {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&buf[idx & mask]), v);
            idx += 2;
        }

    #endif
    #if defined(FISK_HAS_AVX2)

        void consume(__m256i v) noexcept
        {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&buf[idx & mask]), v);
            idx += 4;
        }

    #endif
    #if defined(FISK_HAS_AVX512)

        void consume(__m512i v) noexcept
        {
            _mm512_storeu_si512(reinterpret_cast<void*>(&buf[idx & mask]), v);
            idx += 8;
        }

    #endif
    #if defined(FISK_HAS_NEON)

        void consume(uint64x2_t v) noexcept
        {
            vst1q_u64(&buf[idx & mask], v);
            idx += 2;
        }

    #endif

    // Sums only the slots this call actually wrote (buf[0, idx) when idx never wrapped), not the
    // whole buffer: a shared buffer generously sized relative to one call's touch count (as most
    // callers do, rather than sizing it to fit exactly) would otherwise still hold an earlier
    // call's leftover values past idx, and get summed right along with this call's real ones.
    // Once idx passes buf.size(), every slot has been written by this call at least once, so the
    // whole buffer is this call's own data again.
    std::uint64_t finalize() const noexcept
    {
        std::size_t const n = idx < buf.size() ? idx : buf.size();
        std::uint64_t s = 0;
        for (std::size_t i = 0; i < n; ++i) {
            s += buf[i];
        }
        return s;
    }
};

inline Sink make_sink(std::vector<std::uint64_t>& buf)
{
    return Sink{buf, buf.size() - 1, 0};
}

// =================================================================================================
//     Unknown sink
// =================================================================================================

#else
#error "Unknown FISK_BENCHMARK_SINK_MODE"
#endif
