#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

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
//   network_table   Masks for the algorithm
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
//     SIMD Kernels for Bit Extract Network Table Implementation
// =================================================================================================

// These kernels implement the network table algorithm for bit extraction.
//
// The kernels simply copy the entries of the network table for a given mask into simd registers,
// such that the bit extraction can be applied in parallel across simd lanes.

// -------------------------------------------------------------------------------------------------
//     Scalar Kernel
// -------------------------------------------------------------------------------------------------

/**
 * @brief Kernel for bit extract using a simple scalar implementation of the network table.
 */
struct BitExtractNetworkKernelScalar
{
    // Regular scalar values for the network table
    using simd_vector = std::uint64_t;
    static constexpr int lanes = 1;
    BitExtractNetworkTable network_table;

    BitExtractNetworkKernelScalar( std::uint64_t const mask )
        : network_table( bit_extract_network_table_preprocess( mask ))
    {}

    static simd_vector load( std::uint64_t const* a ) noexcept
    {
        return a[0];
    }

    static void store( simd_vector const v, std::uint64_t* out ) noexcept
    {
        out[0] = v;
    }

    simd_vector bit_extract( simd_vector x ) const noexcept
    {
        return bit_extract_network_table( x, network_table );
    }
};

// -------------------------------------------------------------------------------------------------
//     SSE2 Kernel
// -------------------------------------------------------------------------------------------------

#if defined(HAVE_SSE2)

/**
 * @brief Kernel for bit extract using an SSE2 implementation of the network table.
 */
struct BitExtractNetworkKernelSSE2
{
    // Regular network table values, and simd copies across lanes
    using simd_vector = __m128i;
    static constexpr int lanes = 2;
    BitExtractNetworkTable network_table;
    __m128i mask_simd{};
    __m128i sieves_simd[6]{};

    BitExtractNetworkKernelSSE2( std::uint64_t const mask )
        : network_table( bit_extract_network_table_preprocess( mask ))
    {
        // Copy the scalar mask and table over to the simd vectors.
        static_assert( sizeof(long long) >= sizeof(std::uint64_t) );
        mask_simd = _mm_set1_epi64x( static_cast<long long>( network_table.mask ));
        for( size_t i = 0; i < 6; ++i ) {
            sieves_simd[i] = _mm_set1_epi64x( static_cast<long long>( network_table.sieves[i] ));
        }
    }

    static simd_vector load( std::uint64_t const* a ) noexcept
    {
        return _mm_load_si128(reinterpret_cast<__m128i const*>(a));
        // return _mm_loadu_si128(reinterpret_cast<__m128i const*>(a));
    }

    static void store( simd_vector const v, std::uint64_t* out ) noexcept
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
        return bit_extract_network_table( x, network_table );
    }
};

#endif

// -------------------------------------------------------------------------------------------------
//     AVX2 Kernel
// -------------------------------------------------------------------------------------------------

#if defined(HAVE_AVX2)

/**
 * @brief Kernel for bit extract using an AVX2 implementation of the network table.
 */
struct BitExtractNetworkKernelAVX2
{
    using simd_vector = __m256i;
    static constexpr int lanes = 4;
    BitExtractNetworkTable network_table;
    __m256i mask_simd{};
    __m256i sieves_simd[6]{};

    BitExtractNetworkKernelAVX2( std::uint64_t const mask )
        : network_table( bit_extract_network_table_preprocess( mask ))
    {
        static_assert( sizeof(long long) >= sizeof(std::uint64_t) );
        mask_simd = _mm256_set1_epi64x( static_cast<long long>( network_table.mask ));
        for( size_t i = 0; i < 6; ++i ) {
            sieves_simd[i] = _mm256_set1_epi64x( static_cast<long long>( network_table.sieves[i] ));
        }
    }

    static simd_vector load( std::uint64_t const* a ) noexcept
    {
        return _mm256_load_si256( reinterpret_cast<__m256i const*>( a ));
        // return _mm256_loadu_si256( reinterpret_cast<__m256i const*>( a ));
    }

    static void store( simd_vector const v, std::uint64_t* out ) noexcept
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
        return bit_extract_network_table( x, network_table );
    }
};

#endif

// -------------------------------------------------------------------------------------------------
//     AVX512 Kernel
// -------------------------------------------------------------------------------------------------

#if defined(HAVE_AVX512F)

/**
 * @brief Kernel for bit extract using an AVX512 implementation of the network table.
 */
struct BitExtractNetworkKernelAVX512
{
    using simd_vector = __m512i;
    static constexpr int lanes = 8;
    BitExtractNetworkTable network_table;
    __m512i mask_simd{};
    __m512i sieves_simd[6]{};

    BitExtractNetworkKernelAVX512( std::uint64_t const mask )
        : network_table( bit_extract_network_table_preprocess( mask ))
    {
        static_assert( sizeof(long long) >= sizeof(std::uint64_t) );
        mask_simd = _mm512_set1_epi64( static_cast<long long>( network_table.mask ));
        for( size_t i = 0; i < 6; ++i ) {
            sieves_simd[i] = _mm512_set1_epi64( static_cast<long long>( network_table.sieves[i] ));
        }
    }

    static simd_vector load( std::uint64_t const* a ) noexcept
    {
        return _mm512_load_si512( reinterpret_cast<__m512i const*>( a ));
        // return _mm512_loadu_si512( reinterpret_cast<__m512i const*>( a ));
    }

    static void store( simd_vector const v, std::uint64_t* out ) noexcept
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
        return bit_extract_network_table( x, network_table );
    }
};

#endif

// -------------------------------------------------------------------------------------------------
//     NEON Kernel
// -------------------------------------------------------------------------------------------------

#if defined(HAVE_NEON)

/**
 * @brief Kernel for bit extract using an ARM NEON implementation of the network table.
 */
struct BitExtractNetworkKernelNEON
{
    using simd_vector = uint64x2_t;
    static constexpr int lanes = 2;
    uint64x2_t mask_simd{};
    uint64x2_t sieves_simd[6]{};

    BitExtractNetworkKernelNEON( std::uint64_t const mask )
        : network_table( bit_extract_network_table_preprocess( mask ))
    {
        mask_simd = vdupq_n_u64( network_table.mask );
        for( size_t i = 0; i < 6; ++i) {
            sieves_simd[i] = vdupq_n_u64( network_table.sieves[i] );
        }
    }

    static simd_vector load( std::uint64_t const* a ) noexcept
    {
        return vld1q_u64(a);
    }

    static void store( simd_vector const v, std::uint64_t* out ) noexcept
    {
        vst1q_u64(out, v);
    }

    simd_vector bit_extract( simd_vector x ) const noexcept
    {
        x = vandq_u64(x, mask_simd);
        auto step_ = [&](int s, uint64x2_t mvv)
        {
            uint64x2_t t = vandq_u64(x, mvv);
            x = vorrq_u64(veorq_u64(x, t), vshrq_n_u64(t, s));
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
        return bit_extract_network_table( x, network_table );
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
struct BitExtractBlockKernelScalar
{
    // Regular scalar values for the block table
    using simd_vector = std::uint64_t;
    static constexpr int lanes = 1;
    BitExtractBlockTable block_table;

    BitExtractBlockKernelScalar( std::uint64_t const mask )
        : block_table( bit_extract_block_table_preprocess( mask ))
    {}

    static simd_vector load( std::uint64_t const* a ) noexcept
    {
        return a[0];
    }

    static void store( simd_vector const v, std::uint64_t* out ) noexcept
    {
        out[0] = v;
    }

    simd_vector bit_extract( simd_vector x ) const noexcept
    {
        return bit_extract_block_table( x, block_table );
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
struct BitExtractBlockKernelSSE2
{
    static_assert(UF > 0 && ((UF & (UF - 1)) == 0), "UF must be a power of 2");
    static_assert(UF <= 32, "UF must be <= 32");
    static_assert((32 % UF) == 0, "UF must divide 32");

    using simd_vector = __m128i;
    static constexpr int lanes = 2;

    BitExtractBlockTable block_table;
    __m128i blocks_simd[33]{};
    __m128i shifts_simd[33]{};

    explicit BitExtractBlockKernelSSE2(std::uint64_t const mask)
        : block_table(bit_extract_block_table_preprocess(mask))
    {
        assert( block_table.blocks[32] == 0 && block_table.shifts[32] == 0 );
        static_assert(sizeof(long long) >= sizeof(std::uint64_t));
        for (std::size_t i = 0; i < 33; ++i) {
            // Set the blocks and shift. Shift count in low 64-bit lane (x86 convention)
            auto const mm = static_cast<long long>( block_table.blocks[i] );
            auto const ss = static_cast<long long>( block_table.shifts[i] );
            blocks_simd[i] = _mm_set_epi64x(mm, mm);
            shifts_simd[i] = _mm_set_epi64x(0, ss);
        }
    }

    static simd_vector load(std::uint64_t const* a) noexcept
    {
        return _mm_load_si128(reinterpret_cast<__m128i const*>(a));
        // return _mm_loadu_si128(reinterpret_cast<__m128i const*>(a));
    }

    static void store(simd_vector const v, std::uint64_t* out) noexcept
    {
        _mm_store_si128(reinterpret_cast<__m128i*>(out), v);
        // _mm_storeu_si128(reinterpret_cast<__m128i*>(out), v);
    }

    simd_vector bit_extract(simd_vector x) const noexcept
    {
        simd_vector res = _mm_setzero_si128();
        std::size_t i = 0;

        while (block_table.blocks[i]) {
            simd_vector combined = or_reduce_<0, UF, UF>(x, i);
            res = _mm_or_si128(res, combined);
            i += UF;
        }
        return res;
    }

    std::uint64_t bit_extract(std::uint64_t x) const noexcept
    {
        return bit_extract_block_table(x, block_table);
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
struct BitExtractBlockKernelAVX2
{
    static_assert(UF > 0 && ((UF & (UF - 1)) == 0), "UF must be a power of 2");
    static_assert(UF <= 32, "UF must be <= 32");
    static_assert((32 % UF) == 0, "UF must divide 32");

    using simd_vector = __m256i;
    static constexpr int lanes = 4;

    BitExtractBlockTable block_table;
    __m256i blocks_simd[33]{};
    __m128i shifts_simd[33]{};

    explicit BitExtractBlockKernelAVX2(std::uint64_t const mask)
        : block_table(bit_extract_block_table_preprocess(mask))
    {
        assert( block_table.blocks[32] == 0 && block_table.shifts[32] == 0 );
        static_assert(sizeof(long long) >= sizeof(std::uint64_t));
        for (std::size_t i = 0; i < 33; ++i) {
            blocks_simd[i] = _mm256_set1_epi64x( static_cast<long long>( block_table.blocks[i] ));
            shifts_simd[i] = _mm_set_epi64x( 0,  static_cast<long long>( block_table.shifts[i] ));
        }
    }

    static simd_vector load(std::uint64_t const* a) noexcept
    {
        return _mm256_load_si256(reinterpret_cast<__m256i const*>(a));
        // return _mm256_loadu_si256(reinterpret_cast<__m256i const*>(a));
    }

    static void store(simd_vector const v, std::uint64_t* out) noexcept
    {
        _mm256_store_si256(reinterpret_cast<__m256i*>(out), v);
        // _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), v);
    }

    simd_vector bit_extract(simd_vector x) const noexcept
    {
        simd_vector res = _mm256_setzero_si256();
        std::size_t i = 0;

        while (block_table.blocks[i]) {
            simd_vector combined = or_reduce_<0, UF, UF>(x, i);
            res = _mm256_or_si256(res, combined);
            i += UF;
        }

        return res;
    }

    std::uint64_t bit_extract( std::uint64_t x ) const noexcept
    {
        return bit_extract_block_table( x, block_table );
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
struct BitExtractBlockKernelAVX512
{
    static_assert(UF > 0 && ((UF & (UF - 1)) == 0), "UF must be a power of 2");
    static_assert(UF <= 32, "UF must be <= 32");
    static_assert((32 % UF) == 0, "UF must divide 32");

    using simd_vector = __m512i;
    static constexpr int lanes = 8;

    BitExtractBlockTable block_table;
    __m512i blocks_simd[33]{};
    __m128i shifts_simd[33]{};

    explicit BitExtractBlockKernelAVX512(std::uint64_t const mask)
        : block_table(bit_extract_block_table_preprocess(mask))
    {
        assert( block_table.blocks[32] == 0 && block_table.shifts[32] == 0 );
        static_assert(sizeof(long long) >= sizeof(std::uint64_t));
        for (std::size_t i = 0; i < 33; ++i) {
            blocks_simd[i] = _mm512_set1_epi64(static_cast<long long>( block_table.blocks[i] ));
            shifts_simd[i] = _mm_set_epi64x(0, static_cast<long long>( block_table.shifts[i] ));
        }
    }

    static simd_vector load(std::uint64_t const* a) noexcept
    {
        return _mm512_load_si512( reinterpret_cast<__m512i const*>( a ));
        // return _mm512_loadu_si512( reinterpret_cast<__m512i const*>( a ));
    }

    static void store(simd_vector const v, std::uint64_t* out) noexcept
    {
        _mm512_store_si512( reinterpret_cast<__m512i*>(out), v );
        // _mm512_storeu_si512( reinterpret_cast<__m512i*>(out), v );
    }

    simd_vector bit_extract(simd_vector x) const noexcept
    {
        simd_vector res = _mm512_setzero_si512();
        std::size_t i = 0;

        while (block_table.blocks[i]) {
            simd_vector combined = or_reduce_<0, UF, UF>(x, i);
            res = _mm512_or_si512(res, combined);
            i += UF;
        }
        return res;
    }

    std::uint64_t bit_extract(std::uint64_t x) const noexcept
    {
        return bit_extract_block_table(x, block_table);
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
struct BitExtractBlockKernelNEON
{
    static_assert(UF > 0 && ((UF & (UF - 1)) == 0), "UF must be a power of 2");
    static_assert(UF <= 32, "UF must be <= 32");
    static_assert((32 % UF) == 0, "UF must divide 32");

    using simd_vector = uint64x2_t;
    static constexpr int lanes = 2;

    BitExtractBlockTable block_table;
    uint64x2_t blocks_simd[33]{};
    int64x2_t  shifts_simd[33]{};  // signed: negative = shift right

    explicit BitExtractBlockKernelNEON(std::uint64_t const mask)
        : block_table(bit_extract_block_table_preprocess(mask))
    {
        assert( block_table.blocks[32] == 0 && block_table.shifts[32] == 0 );
        for (std::size_t i = 0; i < 33; ++i) {
            blocks_simd[i] = vdupq_n_u64( block_table.blocks[i] );

            // vshlq_u64 shifts left for positive, right for negative, so store -s in both lanes
            shifts_simd[i] = vdupq_n_s64( -static_cast<std::int64_t>( block_table.shifts[i] ));
        }
    }

    static simd_vector load(std::uint64_t const* a) noexcept
    {
        return vld1q_u64(a);
    }

    static void store(simd_vector const v, std::uint64_t* out) noexcept
    {
        vst1q_u64(out, v);
    }

    simd_vector bit_extract(simd_vector x) const noexcept
    {
        simd_vector res = vdupq_n_u64(0);
        std::size_t i = 0;

        while (block_table.blocks[i]) {
            simd_vector combined = or_reduce_<0, UF, UF>(x, i);
            res = vorrq_u64(res, combined);
            i += UF;
        }
        return res;
    }

    std::uint64_t bit_extract(std::uint64_t x) const noexcept
    {
        return bit_extract_block_table(x, block_table);
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
