#pragma once

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/util/intrinsics.hpp"
#include "fisk/simd/simd.hpp"
#include "fisk/simd/scalar.hpp"

namespace fisk {

// =================================================================================================
//     Kernel Concept
// =================================================================================================

// A kernel implements a bit extraction algorithm for a SIMD arch (simd/simd.hpp), with the tables
// of one mask, and applies it to all lanes of a SIMD vector in parallel. It names that arch as
// `simd_arch`, for functions that fill the vectors of the kernel with operations of the arch.
//
// The in and out arguments of load() and store() are expected to be aligned to the vector size.
// For lane values held in registers, from_values() builds the vector without going through memory.
// Besides bit_extract() on a whole SIMD vector, bit_extract_scalar() applies the algorithm to a
// single uint64_t, such that the caller can process the tail of loop iterations with it.
//
// Overloads or extensions to multiple masks are possible, but as the number of available lanes
// differs, would not be consistent across SIMD archs. We hence focus here on a single mask.

/**
 * @brief Concept for a bit extraction kernel, such as BitExtractKernelButterfly<SimdAVX2>.
 */
template <typename Kernel>
concept BitExtractKernel = requires(
    Kernel const kernel,
    typename Kernel::simd_vector const v,
    std::uint64_t const* in,
    std::uint64_t* out,
    std::uint64_t x,
    std::array<std::uint64_t, Kernel::lanes> values
) {
    { Kernel::lanes } -> std::convertible_to<std::size_t>;
    Kernel::load(in);
    Kernel::store(v, out);
    Kernel::from_values(values);
    kernel.bit_extract(v);
    { kernel.bit_extract_scalar(x) } -> std::same_as<std::uint64_t>;
} && std::constructible_from<Kernel, std::uint64_t> && SimdArch<typename Kernel::simd_arch>;

// =================================================================================================
//     PEXT Kernel
// =================================================================================================

#if defined(FISK_HAS_BMI2)

/**
 * @brief Kernel for bit extract using hardware BMI2 PEXT, on the single lane of SimdScalar.
 *
 * PEXT has no SIMD form, and CPUs with a fast PEXT run at most one per cycle, so that grouping
 * several extractions into a vector of more lanes does not raise its throughput. The kernel hence
 * takes the scalar arch, such that it can be used wherever the kernels of the SIMD archs are.
 */
struct BitExtractKernelPext
{
    using simd_arch = SimdScalar;
    using simd_vector = SimdScalar::simd_vector;
    static constexpr std::size_t lanes = SimdScalar::lanes;
    BitExtractMask mask{};

    BitExtractKernelPext() = default;

    explicit BitExtractKernelPext(std::uint64_t const mask_value) noexcept
        : mask( BitExtractMask( mask_value ))
    {}

    static simd_vector load(std::uint64_t const* a) noexcept
    {
        return SimdScalar::load_aligned(a);
    }

    static void store(simd_vector const& v, std::uint64_t* out) noexcept
    {
        SimdScalar::store_aligned(v, out);
    }

    static simd_vector from_values(std::array<std::uint64_t, lanes> values) noexcept
    {
        return SimdScalar::from_values(values);
    }

    simd_vector bit_extract(simd_vector x) const noexcept
    {
        return bit_extract_pext(x, mask);
    }

    std::uint64_t bit_extract_scalar(std::uint64_t x) const noexcept
    {
        return bit_extract_pext(x, mask);
    }
};

#endif

// =================================================================================================
//     Butterfly Kernel
// =================================================================================================

/**
 * @brief Kernel for bit extract using the butterfly network, for a SIMD arch.
 *
 * The entries of the butterfly mask are broadcast into SIMD vectors, such that the
 * bit extraction can be applied in parallel across lanes.
 */
template <SimdArch Simd>
struct BitExtractKernelButterfly
{
    using simd_arch = Simd;
    using simd_vector = typename Simd::simd_vector;
    static constexpr std::size_t lanes = Simd::lanes;
    BitExtractButterflyMask mask{};
    simd_vector mask_simd{};
    simd_vector sieves_simd[6]{};

    BitExtractKernelButterfly() = default;

    explicit BitExtractKernelButterfly(std::uint64_t const mask_value)
        : mask(make_bit_extract_butterfly_mask(mask_value))
    {
        mask_simd = Simd::broadcast(mask_value);
        for (std::size_t i = 0; i < 6; ++i) {
            sieves_simd[i] = Simd::broadcast(mask.sieves[i]);
        }
    }

    static simd_vector load(std::uint64_t const* a) noexcept
    {
        return Simd::load_aligned(a);
    }

    static void store(simd_vector const& v, std::uint64_t* out) noexcept
    {
        Simd::store_aligned(v, out);
    }

    static simd_vector from_values(std::array<std::uint64_t, lanes> values) noexcept
    {
        return Simd::from_values(values);
    }

    simd_vector bit_extract(simd_vector x) const noexcept
    {
        return bit_extract_(
            x, mask_simd,
            sieves_simd[0], sieves_simd[1], sieves_simd[2],
            sieves_simd[3], sieves_simd[4], sieves_simd[5]
        );
    }

    std::uint64_t bit_extract_scalar(std::uint64_t x) const noexcept
    {
        return bit_extract_butterfly(x, mask);
    }

    /**
     * @brief Same as bit_extract(), but without a kernel instance, broadcasting the given table on
     * each call instead.
     *
     * Meant for a table that is a compile-time constant, see make_bit_extract_butterfly_mask(),
     * whose broadcasts then become constants that the compiler can keep in registers.
     */
    static simd_vector bit_extract(simd_vector x, BitExtractButterflyMask const& table) noexcept
    {
        return bit_extract_(
            x, Simd::broadcast(table.mask),
            Simd::broadcast(table.sieves[0]), Simd::broadcast(table.sieves[1]),
            Simd::broadcast(table.sieves[2]), Simd::broadcast(table.sieves[3]),
            Simd::broadcast(table.sieves[4]), Simd::broadcast(table.sieves[5])
        );
    }

private:

    // The butterfly network, shared by both overloads of bit_extract(). Forced inline, so that
    // passing the vectors by value never becomes an actual call.
    FISK_ALWAYS_INLINE
    static simd_vector bit_extract_(
        simd_vector x, simd_vector mask_vec,
        simd_vector sieve0, simd_vector sieve1, simd_vector sieve2,
        simd_vector sieve3, simd_vector sieve4, simd_vector sieve5
    ) noexcept {
        x = Simd::bitwise_and(x, mask_vec);
        step_< 1>(x, sieve0);
        step_< 2>(x, sieve1);
        step_< 4>(x, sieve2);
        step_< 8>(x, sieve3);
        step_<16>(x, sieve4);
        step_<32>(x, sieve5);
        return x;
    }

    template <int S>
    static void step_(simd_vector& x, simd_vector sieve) noexcept
    {
        simd_vector const t = Simd::bitwise_and(x, sieve);
        x = Simd::bitwise_or(Simd::bitwise_xor(x, t), Simd::template shift_right<S>(t));
    }
};

// =================================================================================================
//     Block Shift Kernel
// =================================================================================================

/**
 * @brief Kernel for bit extract using the unrolled block shift, for a SIMD arch.
 *
 * The kernel is templetized with an unrolling factor `UF`, for loop unrolling at compile time.
 * As here the compiler might not have sufficient optimization to see through the operations,
 * we manually implement a compile-time balanced tree over [0, UF) for the OR operations that
 * combine the different blocks. That is, instead of a long chain of consecutive (and dependent)
 * operations, we use a tree structure, where several operations can occur in parallel,
 * in order to maximize throughput.
 *
 * @tparam UF Unroll factor for compile-time loop unrolling, defaults to 8-fold unrolling.
 */
template <SimdArch Simd, std::size_t UF = 8>
struct BitExtractKernelBlockShift
{
    static_assert(UF > 0 && ((UF & (UF - 1)) == 0), "UF must be a power of 2");
    static_assert(UF <= 32, "UF must be <= 32");
    static_assert((32 % UF) == 0, "UF must divide 32");

    using simd_arch = Simd;
    using simd_vector = typename Simd::simd_vector;
    using shift_count = typename Simd::shift_count;
    static constexpr std::size_t lanes = Simd::lanes;

    BitExtractBlockShiftMask mask{};
    simd_vector blocks_simd[33]{};
    shift_count shifts_simd[33]{};

    BitExtractKernelBlockShift() = default;

    explicit BitExtractKernelBlockShift(std::uint64_t const mask_value)
        : mask(make_bit_extract_block_shift_mask(mask_value))
    {
        assert( mask.blocks[32] == 0 && mask.shifts[32] == 0 );
        for (std::size_t i = 0; i < 33; ++i) {
            blocks_simd[i] = Simd::broadcast(mask.blocks[i]);
            shifts_simd[i] = Simd::make_shift_count(mask.shifts[i]);
        }
    }

    static simd_vector load(std::uint64_t const* a) noexcept
    {
        return Simd::load_aligned(a);
    }

    static void store(simd_vector const& v, std::uint64_t* out) noexcept
    {
        Simd::store_aligned(v, out);
    }

    static simd_vector from_values(std::array<std::uint64_t, lanes> values) noexcept
    {
        return Simd::from_values(values);
    }

    simd_vector bit_extract(simd_vector x) const noexcept
    {
        simd_vector res = Simd::zero();
        std::size_t i = 0;

        while (mask.blocks[i]) {
            simd_vector combined = or_reduce_<0, UF, UF>(x, i);
            res = Simd::bitwise_or(res, combined);
            i += UF;
        }
        return res;
    }

    std::uint64_t bit_extract_scalar(std::uint64_t x) const noexcept
    {
        return bit_extract_block_shift(x, mask);
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
            return Simd::bitwise_or(a, b);
        }
    }

    template<std::size_t K>
    inline simd_vector term_(simd_vector x, std::size_t i) const noexcept
    {
        return Simd::shift_right(
            Simd::bitwise_and(x, blocks_simd[i + K]),
            shifts_simd[i + K]
        );
    }
};

} // namespace fisk
