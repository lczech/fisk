#pragma once

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/core/intrinsics.hpp"
#include "fisk/core/simd.hpp"

namespace fisk {

// =================================================================================================
//     Kernel Concept
// =================================================================================================

// A kernel implements a bit extraction algorithm for a SIMD arch (core/simd.hpp), with the tables
// of one mask, and applies it to all lanes of a SIMD vector in parallel.
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
} && std::constructible_from<Kernel, std::uint64_t>;

// =================================================================================================
//     Pseudo-SIMD Kernels for PEXT
// =================================================================================================

#if defined(FISK_HAS_BMI2)

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

    static simd_vector from_values(simd_vector values) noexcept
    {
        return values;
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

    std::uint64_t bit_extract_scalar(std::uint64_t x) const noexcept
    {
        return bit_extract_pext(x, mask);
    }
};

#endif

// =================================================================================================
//     Butterfly Table Kernel
// =================================================================================================

/**
 * @brief Kernel for bit extract using the butterfly table, for a SIMD arch.
 *
 * The entries of the butterfly table of the mask are broadcast into SIMD vectors, such that the
 * bit extraction can be applied in parallel across lanes.
 */
template <SimdArch Simd>
struct BitExtractKernelButterfly
{
    using simd_vector = typename Simd::simd_vector;
    static constexpr std::size_t lanes = Simd::lanes;
    BitExtractButterflyTable mask{};
    simd_vector mask_simd{};
    simd_vector sieves_simd[6]{};

    BitExtractKernelButterfly() = default;

    explicit BitExtractKernelButterfly(std::uint64_t const mask_value)
        : mask(make_bit_extract_butterfly_table(mask_value))
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
        x = Simd::bitwise_and(x, mask_simd);
        step_< 1>(x, sieves_simd[0]);
        step_< 2>(x, sieves_simd[1]);
        step_< 4>(x, sieves_simd[2]);
        step_< 8>(x, sieves_simd[3]);
        step_<16>(x, sieves_simd[4]);
        step_<32>(x, sieves_simd[5]);
        return x;
    }

    std::uint64_t bit_extract_scalar(std::uint64_t x) const noexcept
    {
        return bit_extract_butterfly_table(x, mask);
    }

private:

    template <int S>
    static void step_(simd_vector& x, simd_vector sieve) noexcept
    {
        simd_vector const t = Simd::bitwise_and(x, sieve);
        x = Simd::bitwise_or(Simd::bitwise_xor(x, t), Simd::template shift_right<S>(t));
    }
};

// =================================================================================================
//     Block Table Kernel
// =================================================================================================

/**
 * @brief Kernel for bit extract using the unrolled block table, for a SIMD arch.
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
struct BitExtractKernelBlock
{
    static_assert(UF > 0 && ((UF & (UF - 1)) == 0), "UF must be a power of 2");
    static_assert(UF <= 32, "UF must be <= 32");
    static_assert((32 % UF) == 0, "UF must divide 32");

    using simd_vector = typename Simd::simd_vector;
    using shift_count = typename Simd::shift_count;
    static constexpr std::size_t lanes = Simd::lanes;

    BitExtractBlockTable mask{};
    simd_vector blocks_simd[33]{};
    shift_count shifts_simd[33]{};

    BitExtractKernelBlock() = default;

    explicit BitExtractKernelBlock(std::uint64_t const mask_value)
        : mask(make_bit_extract_block_table(mask_value))
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
 *     // process with the std::array<MyKernel, N> of kernels
 * });
 * ```
 *
 * For spaced k-mer extraction, see make_spaced_kmer_kernels() for constructing the kernels.
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

} // namespace fisk
