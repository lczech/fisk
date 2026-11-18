#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <type_traits>
#include <vector>

#include "fisk/util/aligned_allocator.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Aligned Allocator Tests
// =================================================================================================

namespace {

bool is_aligned_(void const* p, std::size_t alignment)
{
    return reinterpret_cast<std::uintptr_t>(p) % alignment == 0;
}

} // namespace

TEST(UtilAlignedAllocator, VectorsAndStrings)
{
    // Every size, also after growing, copying and with several allocations alive at once, which
    // the heap would otherwise place at 16 bytes past a cache line as often as not. The strings are
    // longer than the 22 characters that libc++ stores inside the string object itself, unaligned.
    for (std::size_t n = 1; n <= 200; ++n) {
        AlignedVector<std::uint8_t> bytes(n);
        AlignedVector<std::uint64_t> words(n);
        AlignedString str(n + 22, 'A');
        EXPECT_TRUE(is_aligned_(bytes.data(), 64));
        EXPECT_TRUE(is_aligned_(words.data(), 64));
        EXPECT_TRUE(is_aligned_(str.data(), 64));

        bytes.resize(3 * n);
        str.append(n, 'C');
        AlignedVector<std::uint64_t> const copy = words;
        EXPECT_TRUE(is_aligned_(bytes.data(), 64));
        EXPECT_TRUE(is_aligned_(str.data(), 64));
        EXPECT_TRUE(is_aligned_(copy.data(), 64));
        EXPECT_EQ(copy, words);
    }
}

TEST(UtilAlignedAllocator, OtherAlignmentsAndRebind)
{
    std::vector<std::uint16_t, AlignedAllocator<std::uint16_t, 4096>> page(10);
    EXPECT_TRUE(is_aligned_(page.data(), 4096));

    // A container that allocates other types than its elements keeps the alignment.
    using Rebound = std::allocator_traits<AlignedAllocator<char>>::rebind_alloc<std::uint64_t>;
    static_assert(std::is_same_v<Rebound, AlignedAllocator<std::uint64_t>>);
    EXPECT_TRUE(AlignedAllocator<char>{} == AlignedAllocator<std::uint64_t>{});
}

TEST(UtilAlignedAllocator, StringConvertsToStringView)
{
    AlignedString const str = "ACGTNACGT";
    std::string_view const view = str;
    EXPECT_EQ(view, "ACGTNACGT");
    EXPECT_EQ(view.data(), str.data());
}
