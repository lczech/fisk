#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <type_traits>

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
  #include <emmintrin.h>
#endif
#if defined(__AVX2__)
  #include <immintrin.h>
#endif
#if defined(__AVX512F__)
  #include <immintrin.h>
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
  #include <arm_neon.h>
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
//   bit_extract()   Perform the bit extraction on all lanes
//
// Note that we expect the in and out arguments for load() and store() to be properly memory
// aligned. Alternative functions for unaligned loads and stores are provided in comments.
//
// The kernels offer functionality to run bit extraction in parallel across SIMD lanes,
// using the same mask on different values. Overloads or extensions to multiple masks
// are possible, but as the number of available lanes differes, would not be consistent
// across SIMD architectures. We hence focus here on the case with a single mask.

// =================================================================================================
//     SIMD Kernels for Bit Extract Network Table Implementation
// =================================================================================================

// -------------------------------------------------------------------------------------------------
//     Scalar Kernel
// -------------------------------------------------------------------------------------------------

struct BitExtractKernelScalar
{
    // Regular scalar values for the network table
    using simd_vector = std::uint64_t;
    static constexpr int lanes = 1;
    BitExtractNetworkTable network_table;

    BitExtractKernelScalar( std::uint64_t const mask )
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

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
struct BitExtractKernelSSE2
{
    // Regular network table values, and simd copies across lanes
    using simd_vector = __m128i;
    static constexpr int lanes = 2;
    BitExtractNetworkTable network_table;
    __m128i mask_simd{};
    __m128i sieves_simd[6]{};

    BitExtractKernelSSE2( std::uint64_t const mask )
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
};
#endif

// -------------------------------------------------------------------------------------------------
//     AVX2 Kernel
// -------------------------------------------------------------------------------------------------

#if defined(__AVX2__)
struct BitExtractKernelAVX2
{
    using simd_vector = __m256i;
    static constexpr int lanes = 4;
    BitExtractNetworkTable network_table;
    __m256i mask_simd{};
    __m256i sieves_simd[6]{};

    BitExtractKernelAVX2( std::uint64_t const mask )
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
};
#endif

// -------------------------------------------------------------------------------------------------
//     AVX512 Kernel
// -------------------------------------------------------------------------------------------------

#if defined(__AVX512F__)
struct BitExtractKernelAVX512
{
    using simd_vector = __m512i;
    static constexpr int lanes = 8;
    BitExtractNetworkTable network_table;
    __m512i mask_simd{};
    __m512i sieves_simd[6]{};

    BitExtractKernelAVX512( std::uint64_t const mask )
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
        return _mm512_load_si512( reinterpret_cast<__m512i*>( a ));
        // return _mm512_loadu_si512( reinterpret_cast<__m512i*>( a ));
    }

    static void store( simd_vector const v, std::uint64_t* out ) noexcept
    {
        _mm512_store_si512( reinterpret_cast<__m512i*>(out), v );
        // _mm512_storeu_si512( reinterpret_cast<__m512i*>(out), v );
    }

    simd_vector bit_extract( simd_vector x ) const noexcept
    {
        x = _mm512_and_si512(x, mask_simd);
        auto step_ = [&](int s, __m512i mvv)
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
};
#endif

// -------------------------------------------------------------------------------------------------
//     NEON Kernel
// -------------------------------------------------------------------------------------------------

#if defined(__aarch64__) && defined(__ARM_NEON)
struct BitExtractKernelNEON
{
    using simd_vector = uint64x2_t;
    static constexpr int lanes = 2;
    uint64x2_t mask_simd{};
    uint64x2_t sieves_simd[6]{};

    BitExtractKernelNEON( std::uint64_t const mask )
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
};
#endif
