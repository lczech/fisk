#pragma once

#include <cstddef>
#include <limits>
#include <new>
#include <string>
#include <type_traits>
#include <vector>

namespace fisk {

// =================================================================================================
//     Aligned Allocator
// =================================================================================================

/**
 * @brief An allocator for standard containers whose memory starts at a multiple of `Alignment`
 * bytes, 64 by default: a cache line, and the width of an AVX-512 register.
 *
 * The SIMD algorithms load and store whole vectors. On memory that is not aligned to the vector
 * width, every other 32-byte access and every 64-byte access spans two cache lines, which can cost
 * time, mostly with AVX-512. fisk keeps its own buffers aligned with this allocator, such as the
 * storage of PackedSequence. The input sequences are the caller's: any alignment works, and
 * AlignedString or AlignedVector can make them faster to read, see the README.
 *
 * `std::allocator` only aligns to `alignof(T)`, which is at most 16 bytes for the types here, so
 * this allocator uses the aligned `operator new` of C++17 for any type.
 */
template <typename T, std::size_t Alignment = 64>
class AlignedAllocator
{
public:

    static_assert(
        Alignment >= alignof(T) && (Alignment & (Alignment - 1)) == 0,
        "AlignedAllocator: the alignment has to be a power of two, and at least that of T"
    );

    using value_type = T;
    using is_always_equal = std::true_type;

    static constexpr std::size_t alignment = Alignment;

    /**
     * @brief The same allocator for elements of type `U`, as containers need it.
     */
    template <typename U>
    struct rebind
    {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(AlignedAllocator<U, Alignment> const&) noexcept
    {}

    [[nodiscard]] T* allocate(std::size_t n)
    {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{Alignment}));
    }

    // Without the size, as clang before version 19 does not declare the sized operator delete by
    // default.
    void deallocate(T* p, std::size_t) noexcept
    {
        ::operator delete(p, std::align_val_t{Alignment});
    }

    template <typename U>
    bool operator==(AlignedAllocator<U, Alignment> const&) const noexcept
    {
        return true;
    }
};

/**
 * @brief A `std::vector` whose elements start at a multiple of 64 bytes, see AlignedAllocator.
 */
template <typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;

/**
 * @brief A `std::string` whose characters start at a multiple of 64 bytes, see AlignedAllocator.
 *
 * It converts to `std::string_view`, as the ASCII input of fisk takes it. Strings short enough to
 * be stored inside the object itself, at most 15 characters with libstdc++ and MSVC and 22 with
 * libc++, are not aligned. fisk reads any input correctly: the alignment only affects the speed.
 */
using AlignedString = std::basic_string<char, std::char_traits<char>, AlignedAllocator<char>>;

} // namespace fisk
