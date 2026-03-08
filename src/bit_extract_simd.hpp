#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

// Preprocessor checks for intrinsics support. We use our CMake definitions here as a base check,
// but also allow to compile with support for intrinsics if the compiler provides it, even if not
// enabled in CMake, as long as the relevant headers are available. This allows to compile with
// intrinsics support on platforms where we do not explicitly test for it in CMake, but where the
// compiler and hardware support it anyway.
// The definitions are then set for usage in the code below.
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
  #include <emmintrin.h>
  #define HAVE_SSE2 1
#endif
#if defined(__AVX2__)
  #include <immintrin.h>
  #define HAVE_AVX2 1
#endif
#if defined(__AVX512F__)
  #include <immintrin.h>
  #define HAVE_AVX512F 1
#endif
#if (defined(__aarch64__) && defined(__ARM_NEON))
  #include <arm_neon.h>
  #define HAVE_NEON 1
#endif

#include "bit_extract.hpp"
#include "seq_enc.hpp"

// Kernel concept:
//
//   simd_vector     Type of the SIMD vector
//   lanes           Number of SIMD lanes
//   ..._table       Masks for the algorithm
//
//   load()          Load data from external variables into the lanes
//   store()         Store data from lanes into external variables
//   bit_extract()   Perform the bit extraction on all lanes, or a scalar
//
// Note that we expect the in and out arguments for load() and store() to be properly memory
// aligned. Alternative functions for unaligned loads and stores are provided in comments.
//
// The bit_extract() function is always overloaded for both the SIMD vector and a normal scalar
// uint64_t, such that the caller can process the tail of loop iterations with the latter.
//
// The kernels offer functionality to run bit extraction in parallel across SIMD lanes,
// using the same mask on different values. Overloads or extensions to multiple masks
// are possible, but as the number of available lanes differes, would not be consistent
// across SIMD architectures. We hence focus here on the case with a single mask.

// =================================================================================================
//     Pseudo-SIMD Kernels for PEXT
// =================================================================================================

#if defined(HAVE_BMI2)

/**
 * @brief Kernel for bit extract using hardware BMI2 PEXT, evaluated independently across lanes.
 *
 * This is not true SIMD: each lane is processed by a separate `_pext_u64()` call.
 * However, grouping several independent extractions together can still improve throughput
 * by exposing instruction-level parallelism and helping to hide PEXT latency.
 *
 * Valid lane counts are 1, 2, 4, and 8, i.e., how many PEXT are being run.
 */
template <std::size_t Lanes = 8>
struct BitExtractKernelPEXT
{
    static_assert(
        Lanes == 1 || Lanes == 2 || Lanes == 4 || Lanes == 8,
        "BitExtractKernelPEXT only supports 1, 2, 4, or 8 lanes."
    );

    using simd_vector = std::array<std::uint64_t, Lanes>;
    static constexpr std::size_t lanes = Lanes;
    BitExtractMask mask{};

    BitExtractKernelPEXT() = default;

    explicit BitExtractKernelPEXT(std::uint64_t const mask_value) noexcept
        : mask( BitExtractMask( mask_value ))
    {}

    static simd_vector load(std::uint64_t const* a) noexcept
    {
        simd_vector v{};
        for (std::size_t i = 0; i < Lanes; ++i) {
            v[i] = a[i];
        }
        return v;
    }

    static void store(simd_vector const& v, std::uint64_t* out) noexcept
    {
        for (std::size_t i = 0; i < Lanes; ++i) {
            out[i] = v[i];
        }
    }

    simd_vector bit_extract(simd_vector const& x) const noexcept
    {
        simd_vector out{};
        if constexpr (Lanes >= 1) {
            out[0] = bit_extract_pext(x[0], mask);
        }
        if constexpr (Lanes >= 2) {
            out[1] = bit_extract_pext(x[1], mask);
        }
        if constexpr (Lanes >= 4) {
            out[2] = bit_extract_pext(x[2], mask);
            out[3] = bit_extract_pext(x[3], mask);
        }
        if constexpr (Lanes >= 8) {
            out[4] = bit_extract_pext(x[4], mask);
            out[5] = bit_extract_pext(x[5], mask);
            out[6] = bit_extract_pext(x[6], mask);
            out[7] = bit_extract_pext(x[7], mask);
        }
        return out;
    }

    std::uint64_t bit_extract(std::uint64_t x) const noexcept
    {
        return bit_extract_pext(x, mask);
    }
};

#endif

// =================================================================================================
//     SIMD Kernels for Bit Extract Butterfly Table Implementation
// =================================================================================================

// These kernels implement the butterfly table algorithm for bit extraction.
//
// The kernels simply copy the entries of the butterfly table for a given mask into simd registers,
// such that the bit extraction can be applied in parallel across simd lanes.

// -------------------------------------------------------------------------------------------------
//     Scalar Kernel
// -------------------------------------------------------------------------------------------------

/**
 * @brief Kernel for bit extract using a simple scalar implementation of the butterfly table.
 */
struct BitExtractKernelButterflyScalar
{
    // Regular scalar values for the butterfly table
    using simd_vector = std::uint64_t;
    static constexpr std::size_t lanes = 1;
    BitExtractButterflyTable mask{};

    BitExtractKernelButterflyScalar() = default;

    explicit BitExtractKernelButterflyScalar( std::uint64_t const mask_value )
        : mask( bit_extract_butterfly_table_preprocess( mask_value ))
    {}

    static simd_vector load( std::uint64_t const* a ) noexcept
    {
        return a[0];
    }

    static void store( simd_vector const& v, std::uint64_t* out ) noexcept
    {
        out[0] = v;
    }

    simd_vector bit_extract( simd_vector x ) const noexcept
    {
        return bit_extract_butterfly_table( x, mask );
    }
};

// -------------------------------------------------------------------------------------------------
//     SSE2 Kernel
// -------------------------------------------------------------------------------------------------

#if defined(HAVE_SSE2)

/**
 * @brief Kernel for bit extract using an SSE2 implementation of the butterfly table.
 */
struct BitExtractKernelButterflySSE2
{
    // Regular butterfly table values, and simd copies across lanes
    using simd_vector = __m128i;
    static constexpr std::size_t lanes = 2;
    BitExtractButterflyTable mask{};
    __m128i mask_simd{};
    __m128i sieves_simd[6]{};

    BitExtractKernelButterflySSE2() = default;

    explicit BitExtractKernelButterflySSE2( std::uint64_t const mask_value )
        : mask( bit_extract_butterfly_table_preprocess( mask_value ))
    {
        // Copy the scalar mask and table over to the simd vectors.
        static_assert( sizeof(long long) >= sizeof(std::uint64_t) );
        mask_simd = _mm_set1_epi64x( static_cast<long long>( mask_value ));
        for( size_t i = 0; i < 6; ++i ) {
            sieves_simd[i] = _mm_set1_epi64x( static_cast<long long>( mask.sieves[i] ));
        }
    }

    static simd_vector load( std::uint64_t const* a ) noexcept
    {
        return _mm_load_si128(reinterpret_cast<__m128i const*>(a));
        // return _mm_loadu_si128(reinterpret_cast<__m128i const*>(a));
    }

    static void store( simd_vector const& v, std::uint64_t* out ) noexcept
    {
        _mm_store_si128( reinterpret_cast<__m128i*>(out), v );
        // _mm_storeu_si128( reinterpret_cast<__m128i*>(out), v );
    }

    simd_vector bit_extract( simd_vector x ) const noexcept
    {
        x = _mm_and_si128(x, mask_simd);
        auto step_ = [&](int s, __m128i mvv)
        {
            __m128i t = _mm_and_si128(x, mvv);
            x = _mm_or_si128(_mm_xor_si128(x, t), _mm_srli_epi64(t, s));
        };
        step_(  1, sieves_simd[0] );
        step_(  2, sieves_simd[1] );
        step_(  4, sieves_simd[2] );
        step_(  8, sieves_simd[3] );
        step_( 16, sieves_simd[4] );
        step_( 32, sieves_simd[5] );
        return x;
    }

    std::uint64_t bit_extract( std::uint64_t x ) const noexcept
    {
        return bit_extract_butterfly_table( x, mask );
    }
};

#endif

// -------------------------------------------------------------------------------------------------
//     AVX2 Kernel
// -------------------------------------------------------------------------------------------------

#if defined(HAVE_AVX2)

/**
 * @brief Kernel for bit extract using an AVX2 implementation of the butterfly table.
 */
struct BitExtractKernelButterflyAVX2
{
    using simd_vector = __m256i;
    static constexpr std::size_t lanes = 4;
    BitExtractButterflyTable mask{};
    __m256i mask_simd{};
    __m256i sieves_simd[6]{};

    BitExtractKernelButterflyAVX2() = default;

    explicit BitExtractKernelButterflyAVX2( std::uint64_t const mask_value )
        : mask( bit_extract_butterfly_table_preprocess( mask_value ))
    {
        static_assert( sizeof(long long) >= sizeof(std::uint64_t) );
        mask_simd = _mm256_set1_epi64x( static_cast<long long>( mask_value ));
        for( size_t i = 0; i < 6; ++i ) {
            sieves_simd[i] = _mm256_set1_epi64x( static_cast<long long>( mask.sieves[i] ));
        }
    }

    static simd_vector load( std::uint64_t const* a ) noexcept
    {
        return _mm256_load_si256( reinterpret_cast<__m256i const*>( a ));
        // return _mm256_loadu_si256( reinterpret_cast<__m256i const*>( a ));
    }

    static void store( simd_vector const& v, std::uint64_t* out ) noexcept
    {
        _mm256_store_si256( reinterpret_cast<__m256i*>(out), v );
        // _mm256_storeu_si256( reinterpret_cast<__m256i*>(out), v );
    }

    simd_vector bit_extract( simd_vector x ) const noexcept
    {
        x = _mm256_and_si256(x, mask_simd);
        auto step_ = [&](int s, __m256i mvv)
        {
            __m256i t = _mm256_and_si256(x, mvv);
            x = _mm256_or_si256(_mm256_xor_si256(x, t), _mm256_srli_epi64(t, s));
        };
        step_(  1, sieves_simd[0] );
        step_(  2, sieves_simd[1] );
        step_(  4, sieves_simd[2] );
        step_(  8, sieves_simd[3] );
        step_( 16, sieves_simd[4] );
        step_( 32, sieves_simd[5] );
        return x;
    }

    std::uint64_t bit_extract( std::uint64_t x ) const noexcept
    {
        return bit_extract_butterfly_table( x, mask );
    }
};

#endif

// -------------------------------------------------------------------------------------------------
//     AVX512 Kernel
// -------------------------------------------------------------------------------------------------

#if defined(HAVE_AVX512F)

/**
 * @brief Kernel for bit extract using an AVX512 implementation of the butterfly table.
 */
struct BitExtractKernelButterflyAVX512
{
    using simd_vector = __m512i;
    static constexpr std::size_t lanes = 8;
    BitExtractButterflyTable mask{};
    __m512i mask_simd{};
    __m512i sieves_simd[6]{};

    BitExtractKernelButterflyAVX512() = default;

    explicit BitExtractKernelButterflyAVX512( std::uint64_t const mask_value )
        : mask( bit_extract_butterfly_table_preprocess( mask_value ))
    {
        static_assert( sizeof(long long) >= sizeof(std::uint64_t) );
        mask_simd = _mm512_set1_epi64( static_cast<long long>( mask_value ));
        for( size_t i = 0; i < 6; ++i ) {
            sieves_simd[i] = _mm512_set1_epi64( static_cast<long long>( mask.sieves[i] ));
        }
    }

    static simd_vector load( std::uint64_t const* a ) noexcept
    {
        return _mm512_load_si512( reinterpret_cast<__m512i const*>( a ));
        // return _mm512_loadu_si512( reinterpret_cast<__m512i const*>( a ));
    }

    static void store( simd_vector const& v, std::uint64_t* out ) noexcept
    {
        _mm512_store_si512( reinterpret_cast<__m512i*>(out), v );
        // _mm512_storeu_si512( reinterpret_cast<__m512i*>(out), v );
    }

    simd_vector bit_extract( simd_vector x ) const noexcept
    {
        x = _mm512_and_si512(x, mask_simd);
        auto step_ = [&](unsigned int s, __m512i mvv)
        {
            __m512i t = _mm512_and_si512(x, mvv);
            x = _mm512_or_si512(_mm512_xor_si512(x, t), _mm512_srli_epi64(t, s));
        };
        step_(  1, sieves_simd[0] );
        step_(  2, sieves_simd[1] );
        step_(  4, sieves_simd[2] );
        step_(  8, sieves_simd[3] );
        step_( 16, sieves_simd[4] );
        step_( 32, sieves_simd[5] );
        return x;
    }

    std::uint64_t bit_extract( std::uint64_t x ) const noexcept
    {
        return bit_extract_butterfly_table( x, mask );
    }
};

#endif

// -------------------------------------------------------------------------------------------------
//     NEON Kernel
// -------------------------------------------------------------------------------------------------

#if defined(HAVE_NEON)

/**
 * @brief Kernel for bit extract using an ARM NEON implementation of the butterfly table.
 */
struct BitExtractKernelButterflyNEON
{
    using simd_vector = uint64x2_t;
    static constexpr std::size_t lanes = 2;
    BitExtractButterflyTable mask{};
    uint64x2_t mask_simd{};
    uint64x2_t sieves_simd[6]{};

    BitExtractKernelButterflyNEON() = default;

    explicit BitExtractKernelButterflyNEON( std::uint64_t const mask_value )
        : mask( bit_extract_butterfly_table_preprocess( mask_value ))
    {
        mask_simd = vdupq_n_u64( mask.mask );
        for( size_t i = 0; i < 6; ++i) {
            sieves_simd[i] = vdupq_n_u64( mask.sieves[i] );
        }
    }

    static simd_vector load( std::uint64_t const* a ) noexcept
    {
        return vld1q_u64(a);
    }

    static void store( simd_vector const& v, std::uint64_t* out ) noexcept
    {
        vst1q_u64(out, v);
    }

    simd_vector bit_extract( simd_vector x ) const noexcept
    {
        // vshrq_n_u64() needs a compile time constant shift value,
        // which we provide here via template paramater
        x = vandq_u64(x, mask_simd);
        step_< 1>(x, sieves_simd[0]);
        step_< 2>(x, sieves_simd[1]);
        step_< 4>(x, sieves_simd[2]);
        step_< 8>(x, sieves_simd[3]);
        step_<16>(x, sieves_simd[4]);
        step_<32>(x, sieves_simd[5]);
        return x;
    }

    std::uint64_t bit_extract( std::uint64_t x ) const noexcept
    {
        return bit_extract_butterfly_table( x, mask );
    }

private:

    template <int Shift>
    inline void step_(simd_vector& x, uint64x2_t mvv) noexcept
    {
        uint64x2_t t = vandq_u64(x, mvv);
        x = vorrq_u64(veorq_u64(x, t), vshrq_n_u64(t, Shift));
    }

};

#endif

// =================================================================================================
//     SIMD Kernels for Bit Extract Unrolled Block Table Implementation
// =================================================================================================

// These kernels implement the block table algorithm for bit extraction.
//
// All kernels are templetized with an unrolling factor `UF`, for loop unrolling at compile time.
// As here the compiler might not have sufficient optimization to see through the operations,
// we manually implement a compile-time balanced tree over [0, UF) for the OR operations that
// combine the different blocks. That is, instead of a long chain of consecutive (and dependent)
// operations, we use a tree structure, where several operations can occur in parallel,
// in order to maximize throughput.

// -------------------------------------------------------------------------------------------------
//     Scalar Kernel
// -------------------------------------------------------------------------------------------------

/**
 * @brief Kernel for bit extract using a simple scalar implementation of the block table.
 */
template<std::size_t UF = 8>
struct BitExtractKernelBlockScalar
{
    static_assert(
        UF == 1 || UF == 2 || UF == 4 || UF == 8 || UF == 16 || UF == 32,
        "Supported unroll factors are 1, 2, 4, 8, 16, 32."
    );
    using simd_vector = std::uint64_t;
    static constexpr std::size_t lanes = 1;
    BitExtractBlockTable mask{};

    BitExtractKernelBlockScalar() = default;

    explicit BitExtractKernelBlockScalar(std::uint64_t mask_value)
        : mask(bit_extract_block_table_preprocess(mask_value))
    {}

    static simd_vector load(std::uint64_t const* a) noexcept
    {
        return a[0];
    }

    static void store(simd_vector const& v, std::uint64_t* out) noexcept
    {
        out[0] = v;
    }

    simd_vector bit_extract(simd_vector x) const noexcept
    {
        return bit_extract_block_table_unrolled<UF>(x, mask);
    }
};

// -------------------------------------------------------------------------------------------------
//     SSE2 Kernel
// -------------------------------------------------------------------------------------------------

#if defined(HAVE_SSE2)

/**
 * @brief Kernel for bit extract using an SSE2 implementation of the block table.
 *
 * @tparam UF Unroll factor for compile-time loop unrolling, defaults to 8-fold unrolling.
 */
template<std::size_t UF = 8>
struct BitExtractKernelBlockSSE2
{
    static_assert(UF > 0 && ((UF & (UF - 1)) == 0), "UF must be a power of 2");
    static_assert(UF <= 32, "UF must be <= 32");
    static_assert((32 % UF) == 0, "UF must divide 32");

    using simd_vector = __m128i;
    static constexpr std::size_t lanes = 2;

    BitExtractBlockTable mask{};
    __m128i blocks_simd[33]{};
    __m128i shifts_simd[33]{};

    BitExtractKernelBlockSSE2() = default;

    explicit BitExtractKernelBlockSSE2(std::uint64_t const mask_value)
        : mask(bit_extract_block_table_preprocess(mask_value))
    {
        assert( mask.blocks[32] == 0 && mask.shifts[32] == 0 );
        static_assert(sizeof(long long) >= sizeof(std::uint64_t));
        for (std::size_t i = 0; i < 33; ++i) {
            // Set the blocks and shift. Shift count in low 64-bit lane (x86 convention)
            auto const mm = static_cast<long long>( mask.blocks[i] );
            auto const ss = static_cast<long long>( mask.shifts[i] );
            blocks_simd[i] = _mm_set_epi64x(mm, mm);
            shifts_simd[i] = _mm_set_epi64x(0, ss);
        }
    }

    static simd_vector load(std::uint64_t const* a) noexcept
    {
        return _mm_load_si128(reinterpret_cast<__m128i const*>(a));
        // return _mm_loadu_si128(reinterpret_cast<__m128i const*>(a));
    }

    static void store(simd_vector const& v, std::uint64_t* out) noexcept
    {
        _mm_store_si128(reinterpret_cast<__m128i*>(out), v);
        // _mm_storeu_si128(reinterpret_cast<__m128i*>(out), v);
    }

    simd_vector bit_extract(simd_vector x) const noexcept
    {
        simd_vector res = _mm_setzero_si128();
        std::size_t i = 0;

        while (mask.blocks[i]) {
            simd_vector combined = or_reduce_<0, UF, UF>(x, i);
            res = _mm_or_si128(res, combined);
            i += UF;
        }
        return res;
    }

    std::uint64_t bit_extract(std::uint64_t x) const noexcept
    {
        return bit_extract_block_table(x, mask);
    }

private:
    template<std::size_t L, std::size_t R, std::size_t U>
    inline simd_vector or_reduce_(simd_vector x, std::size_t i) const noexcept
    {
        static_assert(L < R, "bad bounds");
        if constexpr (R - L == 1) {
            return term_<L>(x, i);
        } else {
            constexpr std::size_t M = L + (R - L) / 2;
            simd_vector a = or_reduce_<L, M, U>(x, i);
            simd_vector b = or_reduce_<M, R, U>(x, i);
            return _mm_or_si128(a, b);
        }
    }

    template<std::size_t K>
    inline simd_vector term_(simd_vector x, std::size_t i) const noexcept
    {
        return _mm_srl_epi64(
            _mm_and_si128(x, blocks_simd[i + K]),
            shifts_simd[i + K]
        );
    }
};

#endif

// -------------------------------------------------------------------------------------------------
//     AVX2 Kernel
// -------------------------------------------------------------------------------------------------

#if defined(HAVE_AVX2)

/**
 * @brief Kernel for bit extract using an AVX2 implementation of the block table.
 *
 * @tparam UF Unroll factor for compile-time loop unrolling, defaults to 8-fold unrolling.
 */
template<std::size_t UF = 8>
struct BitExtractKernelBlockAVX2
{
    static_assert(UF > 0 && ((UF & (UF - 1)) == 0), "UF must be a power of 2");
    static_assert(UF <= 32, "UF must be <= 32");
    static_assert((32 % UF) == 0, "UF must divide 32");

    using simd_vector = __m256i;
    static constexpr std::size_t lanes = 4;

    BitExtractBlockTable mask{};
    __m256i blocks_simd[33]{};
    __m128i shifts_simd[33]{};

    BitExtractKernelBlockAVX2() = default;

    explicit BitExtractKernelBlockAVX2(std::uint64_t const mask_value)
        : mask(bit_extract_block_table_preprocess(mask_value))
    {
        assert( mask.blocks[32] == 0 && mask.shifts[32] == 0 );
        static_assert(sizeof(long long) >= sizeof(std::uint64_t));
        for (std::size_t i = 0; i < 33; ++i) {
            blocks_simd[i] = _mm256_set1_epi64x( static_cast<long long>( mask.blocks[i] ));
            shifts_simd[i] = _mm_set_epi64x( 0,  static_cast<long long>( mask.shifts[i] ));
        }
    }

    static simd_vector load(std::uint64_t const* a) noexcept
    {
        return _mm256_load_si256(reinterpret_cast<__m256i const*>(a));
        // return _mm256_loadu_si256(reinterpret_cast<__m256i const*>(a));
    }

    static void store(simd_vector const& v, std::uint64_t* out) noexcept
    {
        _mm256_store_si256(reinterpret_cast<__m256i*>(out), v);
        // _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), v);
    }

    simd_vector bit_extract(simd_vector x) const noexcept
    {
        simd_vector res = _mm256_setzero_si256();
        std::size_t i = 0;

        while (mask.blocks[i]) {
            simd_vector combined = or_reduce_<0, UF, UF>(x, i);
            res = _mm256_or_si256(res, combined);
            i += UF;
        }

        return res;
    }

    std::uint64_t bit_extract( std::uint64_t x ) const noexcept
    {
        return bit_extract_block_table( x, mask );
    }

private:

    template<std::size_t L, std::size_t R, std::size_t U>
    inline simd_vector or_reduce_(simd_vector x, std::size_t i) const noexcept
    {
        static_assert(L < R, "bad bounds");
        if constexpr (R - L == 1) {
            return term_<L>(x, i);
        } else {
            constexpr std::size_t M = L + (R - L) / 2;
            simd_vector a = or_reduce_<L, M, U>(x, i);
            simd_vector b = or_reduce_<M, R, U>(x, i);
            return _mm256_or_si256(a, b);
        }
    }

    template<std::size_t K>
    inline simd_vector term_(simd_vector x, std::size_t i) const noexcept
    {
        return _mm256_srl_epi64(
            _mm256_and_si256(x, blocks_simd[i + K]),
            shifts_simd[i + K]
        );
    }
};

#endif

// -------------------------------------------------------------------------------------------------
//     AVX512 Kernel
// -------------------------------------------------------------------------------------------------

#if defined(HAVE_AVX512F)

/**
 * @brief Kernel for bit extract using an AVX512 implementation of the block table.
 *
 * @tparam UF Unroll factor for compile-time loop unrolling, defaults to 8-fold unrolling.
 */
template<std::size_t UF = 8>
struct BitExtractKernelBlockAVX512
{
    static_assert(UF > 0 && ((UF & (UF - 1)) == 0), "UF must be a power of 2");
    static_assert(UF <= 32, "UF must be <= 32");
    static_assert((32 % UF) == 0, "UF must divide 32");

    using simd_vector = __m512i;
    static constexpr std::size_t lanes = 8;

    BitExtractBlockTable mask{};
    __m512i blocks_simd[33]{};
    __m128i shifts_simd[33]{};

    BitExtractKernelBlockAVX512() = default;

    explicit BitExtractKernelBlockAVX512(std::uint64_t const mask_value)
        : mask(bit_extract_block_table_preprocess(mask_value))
    {
        assert( mask.blocks[32] == 0 && mask.shifts[32] == 0 );
        static_assert(sizeof(long long) >= sizeof(std::uint64_t));
        for (std::size_t i = 0; i < 33; ++i) {
            blocks_simd[i] = _mm512_set1_epi64(static_cast<long long>( mask.blocks[i] ));
            shifts_simd[i] = _mm_set_epi64x(0, static_cast<long long>( mask.shifts[i] ));
        }
    }

    static simd_vector load(std::uint64_t const* a) noexcept
    {
        return _mm512_load_si512( reinterpret_cast<__m512i const*>( a ));
        // return _mm512_loadu_si512( reinterpret_cast<__m512i const*>( a ));
    }

    static void store(simd_vector const& v, std::uint64_t* out) noexcept
    {
        _mm512_store_si512( reinterpret_cast<__m512i*>(out), v );
        // _mm512_storeu_si512( reinterpret_cast<__m512i*>(out), v );
    }

    simd_vector bit_extract(simd_vector x) const noexcept
    {
        simd_vector res = _mm512_setzero_si512();
        std::size_t i = 0;

        while (mask.blocks[i]) {
            simd_vector combined = or_reduce_<0, UF, UF>(x, i);
            res = _mm512_or_si512(res, combined);
            i += UF;
        }
        return res;
    }

    std::uint64_t bit_extract(std::uint64_t x) const noexcept
    {
        return bit_extract_block_table(x, mask);
    }

private:
    template<std::size_t L, std::size_t R, std::size_t U>
    inline simd_vector or_reduce_(simd_vector x, std::size_t i) const noexcept
    {
        static_assert(L < R, "bad bounds");
        if constexpr (R - L == 1) {
            return term_<L>(x, i);
        } else {
            constexpr std::size_t M = L + (R - L) / 2;
            simd_vector a = or_reduce_<L, M, U>(x, i);
            simd_vector b = or_reduce_<M, R, U>(x, i);
            return _mm512_or_si512(a, b);
        }
    }

    template<std::size_t K>
    inline simd_vector term_(simd_vector x, std::size_t i) const noexcept
    {
        return _mm512_srl_epi64(
            _mm512_and_si512(x, blocks_simd[i + K]),
            shifts_simd[i + K]
        );
    }
};

#endif

// -------------------------------------------------------------------------------------------------
//     NEON Kernel
// -------------------------------------------------------------------------------------------------

#if defined(HAVE_NEON)

/**
 * @brief Kernel for bit extract using an ARM NEON implementation of the block table.
 *
 * @tparam UF Unroll factor for compile-time loop unrolling, defaults to 8-fold unrolling.
 */
template<std::size_t UF = 8>
struct BitExtractKernelBlockNEON
{
    static_assert(UF > 0 && ((UF & (UF - 1)) == 0), "UF must be a power of 2");
    static_assert(UF <= 32, "UF must be <= 32");
    static_assert((32 % UF) == 0, "UF must divide 32");

    using simd_vector = uint64x2_t;
    static constexpr std::size_t lanes = 2;

    BitExtractBlockTable mask{};
    uint64x2_t blocks_simd[33]{};
    int64x2_t  shifts_simd[33]{};  // signed: negative = shift right

    BitExtractKernelBlockNEON() = default;

    explicit BitExtractKernelBlockNEON(std::uint64_t const mask_value)
        : mask(bit_extract_block_table_preprocess(mask_value))
    {
        assert( mask.blocks[32] == 0 && mask.shifts[32] == 0 );
        for (std::size_t i = 0; i < 33; ++i) {
            blocks_simd[i] = vdupq_n_u64( mask.blocks[i] );

            // vshlq_u64 shifts left for positive, right for negative, so store -s in both lanes
            shifts_simd[i] = vdupq_n_s64( -static_cast<std::int64_t>( mask.shifts[i] ));
        }
    }

    static simd_vector load(std::uint64_t const* a) noexcept
    {
        return vld1q_u64(a);
    }

    static void store(simd_vector const& v, std::uint64_t* out) noexcept
    {
        vst1q_u64(out, v);
    }

    simd_vector bit_extract(simd_vector x) const noexcept
    {
        simd_vector res = vdupq_n_u64(0);
        std::size_t i = 0;

        while (mask.blocks[i]) {
            simd_vector combined = or_reduce_<0, UF, UF>(x, i);
            res = vorrq_u64(res, combined);
            i += UF;
        }
        return res;
    }

    std::uint64_t bit_extract(std::uint64_t x) const noexcept
    {
        return bit_extract_block_table(x, mask);
    }

private:
    template<std::size_t L, std::size_t R, std::size_t U>
    inline simd_vector or_reduce_(simd_vector x, std::size_t i) const noexcept
    {
        static_assert(L < R, "bad bounds");
        if constexpr (R - L == 1) {
            return term_<L>(x, i);
        } else {
            constexpr std::size_t M = L + (R - L) / 2;
            simd_vector a = or_reduce_<L, M, U>(x, i);
            simd_vector b = or_reduce_<M, R, U>(x, i);
            return vorrq_u64(a, b);
        }
    }

    template<std::size_t K>
    inline simd_vector term_(simd_vector x, std::size_t i) const noexcept
    {
        // right shift by variable count: vshlq_u64 with negative shift
        return vshlq_u64(
            vandq_u64(x, blocks_simd[i + K]),
            shifts_simd[i + K]
        );
    }
};

#endif

// =================================================================================================
//     SIMD Kernel Dispatcher
// =================================================================================================

/**
 * @brief Runtime-configurable dispatcher for applying multiple bit-extraction kernels with
 * compile-time mask count.
 *
 * This class bridges the gap between a runtime list of bit-extraction masks (e.g., provided as
 * `std::vector<uint64_t>`) and performance-critical code that benefits from knowing the number
 * of masks at compile time. Internally, the dispatcher converts the runtime vector into one of
 * several `std::array<Kernel, N>` alternatives (for `N` in `[1..16]`) stored inside a
 * `std::variant`. The arrays allow downstream algorithms to unroll loops over masks and
 * generate optimal code, for up to 16 masks.
 *
 * Typical usage:
 *
 * ```
 * std::vector<uint64_t> masks = {...};
 * BitExtractKernelDispatcher<MyKernel> dispatcher(masks);
 *
 * dispatcher.run([&](auto const& kernels) {
 *     for_each_spaced_kmer_simd(seq, span_k, kernels,
 *         [&](size_t mask_idx, size_t pos, uint64_t value) {
 *             // consume spaced k-mer
 *         }
 *     );
 * });
 * ```
 *
 * This class hence is merely an optimization to obtain performance through compile time knowledge.
 * It is instead perfectly possible to just store the masks directly in a `std::vector` or such,
 * and use them without any additional abstraction.
 *
 * @tparam Kernel Bit-extraction kernel type constructed from a mask.
 */
template <class Kernel>
class BitExtractKernelDispatcher
{
public:

    // --------------------------------------------------------------------
    //     Constructors
    // --------------------------------------------------------------------

    /**
     * @brief Construct from pre-built kernels.
     */
    explicit BitExtractKernelDispatcher(std::vector<Kernel> const& kernels)
    {
        kernels_ = build_storage_from_kernels_(kernels);
    }

    /**
     * @brief Construct from raw masks of type `std::uint64_t`.
     */
    explicit BitExtractKernelDispatcher(std::vector<std::uint64_t> const& masks)
    {
        kernels_ = build_storage_from_masks_(masks);
    }

    // --------------------------------------------------------------------
    //     Public members
    // --------------------------------------------------------------------

    /**
     * @brief Call a user-provided function with the internally stored fixed-size kernel array.
     *
     * This dispatcher stores the kernels in a std::variant of std::array<Kernel, N>
     * for N in [1..16]. This run() function selects the active alternative and invokes `f`
     * exactly once with a const reference to the active kernel array
     * (`std::array<Kernel, N> const&`).
     *
     * `F` must be invocable with a single argument of type `std::array<Kernel, N> const&`
     * for all N in [1..16], because std::visit requires the visitor to be valid for every
     * alternative. The return type of @p f is forwarded back to the caller.
     */
    template <class F>
    decltype(auto) run(F&& f) const
    {
        return std::visit([&](auto const& arr) -> decltype(auto) {
            return std::forward<F>(f)(arr);
        }, kernels_);
    }

    /**
     * @brief Get the number of kernels (masks) stored.
     */
    std::size_t size() const noexcept
    {
        // Visit the array that is actually stored in the variant, and get its size.
        return std::visit([](auto const& arr) { return arr.size(); }, kernels_);
    }

    // --------------------------------------------------------------------
    //     Internal members
    // --------------------------------------------------------------------

private:

    template <std::size_t N>
    using KernelArray = std::array<Kernel, N>;

    // Storage for 1..16 (default). If another max value is needed, extend this list accordingly.
    using Storage = std::variant<
        KernelArray<1>,  KernelArray<2>,  KernelArray<3>,  KernelArray<4>,
        KernelArray<5>,  KernelArray<6>,  KernelArray<7>,  KernelArray<8>,
        KernelArray<9>,  KernelArray<10>, KernelArray<11>, KernelArray<12>,
        KernelArray<13>, KernelArray<14>, KernelArray<15>, KernelArray<16>
    >;

    // -------------------------------------------
    //     Build from kernels
    // -------------------------------------------

    template <std::size_t N>
    static KernelArray<N> to_array_from_kernels_(const std::vector<Kernel>& vk)
    {
        // Copy over the kernels to the fixed size array.
        KernelArray<N> a{};
        for (std::size_t i = 0; i < N; ++i) {
            a[i] = vk[i];
        }
        return a;
    }

    static Storage build_storage_from_kernels_(std::vector<Kernel> const& kernels)
    {
        std::size_t const n = kernels.size();
        if (n == 0 || n > 16) {
            throw std::invalid_argument("Need 1..16 masks");
        }

        switch (n) {
            case 1:  return Storage{ to_array_from_kernels_<1>(  kernels )};
            case 2:  return Storage{ to_array_from_kernels_<2>(  kernels )};
            case 3:  return Storage{ to_array_from_kernels_<3>(  kernels )};
            case 4:  return Storage{ to_array_from_kernels_<4>(  kernels )};
            case 5:  return Storage{ to_array_from_kernels_<5>(  kernels )};
            case 6:  return Storage{ to_array_from_kernels_<6>(  kernels )};
            case 7:  return Storage{ to_array_from_kernels_<7>(  kernels )};
            case 8:  return Storage{ to_array_from_kernels_<8>(  kernels )};
            case 9:  return Storage{ to_array_from_kernels_<9>(  kernels )};
            case 10: return Storage{ to_array_from_kernels_<10>( kernels )};
            case 11: return Storage{ to_array_from_kernels_<11>( kernels )};
            case 12: return Storage{ to_array_from_kernels_<12>( kernels )};
            case 13: return Storage{ to_array_from_kernels_<13>( kernels )};
            case 14: return Storage{ to_array_from_kernels_<14>( kernels )};
            case 15: return Storage{ to_array_from_kernels_<15>( kernels )};
            case 16: return Storage{ to_array_from_kernels_<16>( kernels )};
            default: throw std::logic_error("unreachable");
        }
    }

    // -------------------------------------------
    //     Build from masks
    // -------------------------------------------

    template <std::size_t N>
    static KernelArray<N> to_array_from_masks_(const std::vector<std::uint64_t>& masks)
    {
        // Directly construct the kernel from the mask.
        // All our kernels take uint64 as constructor argument.
        KernelArray<N> a{};
        for (std::size_t i = 0; i < N; ++i) {
            a[i] = Kernel{masks[i]};
        }
        return a;
    }

    static Storage build_storage_from_masks_(std::vector<std::uint64_t> const& masks)
    {
        std::size_t const n = masks.size();
        if (n == 0 || n > 16) {
            throw std::invalid_argument("Need 1..16 masks");
        }

        switch (n) {
            case 1:  return Storage{ to_array_from_masks_<1>(  masks )};
            case 2:  return Storage{ to_array_from_masks_<2>(  masks )};
            case 3:  return Storage{ to_array_from_masks_<3>(  masks )};
            case 4:  return Storage{ to_array_from_masks_<4>(  masks )};
            case 5:  return Storage{ to_array_from_masks_<5>(  masks )};
            case 6:  return Storage{ to_array_from_masks_<6>(  masks )};
            case 7:  return Storage{ to_array_from_masks_<7>(  masks )};
            case 8:  return Storage{ to_array_from_masks_<8>(  masks )};
            case 9:  return Storage{ to_array_from_masks_<9>(  masks )};
            case 10: return Storage{ to_array_from_masks_<10>( masks )};
            case 11: return Storage{ to_array_from_masks_<11>( masks )};
            case 12: return Storage{ to_array_from_masks_<12>( masks )};
            case 13: return Storage{ to_array_from_masks_<13>( masks )};
            case 14: return Storage{ to_array_from_masks_<14>( masks )};
            case 15: return Storage{ to_array_from_masks_<15>( masks )};
            case 16: return Storage{ to_array_from_masks_<16>( masks )};
            default: throw std::logic_error("unreachable");
        }
    }

    // --------------------------------------------------------------------
    //     Data members
    // --------------------------------------------------------------------

    Storage kernels_;

};
