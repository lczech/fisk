#include <cstdint>

#include "fisk/util/intrinsics.hpp"
#include "fisk/util/random.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Intrinsics Tests
// =================================================================================================

TEST(UtilIntrinsics, ByteSwap16KnownValues)
{
    EXPECT_EQ(byte_swap_16(0x0000), 0x0000);
    EXPECT_EQ(byte_swap_16(0x0001), 0x0100);
    EXPECT_EQ(byte_swap_16(0x0123), 0x2301);
    EXPECT_EQ(byte_swap_16(0xffff), 0xffff);
}

TEST(UtilIntrinsics, ByteSwap16IsInvolution)
{
    // byte_swap_16() applied twice must be the identity, for any input.
    Splitmix64 rng(2024);
    for (int i = 0; i < 1000; ++i) {
        std::uint16_t const x = static_cast<std::uint16_t>(rng.get_uint64());
        EXPECT_EQ(byte_swap_16(byte_swap_16(x)), x);
    }
}

TEST(UtilIntrinsics, ByteSwap32KnownValues)
{
    EXPECT_EQ(byte_swap_32(0x00000000u), 0x00000000u);
    EXPECT_EQ(byte_swap_32(0x00000001u), 0x01000000u);
    EXPECT_EQ(byte_swap_32(0x01234567u), 0x67452301u);
    EXPECT_EQ(byte_swap_32(0xffffffffu), 0xffffffffu);
}

TEST(UtilIntrinsics, ByteSwap32IsInvolution)
{
    // byte_swap_32() applied twice must be the identity, for any input.
    Splitmix64 rng(2024);
    for (int i = 0; i < 1000; ++i) {
        std::uint32_t const x = static_cast<std::uint32_t>(rng.get_uint64());
        EXPECT_EQ(byte_swap_32(byte_swap_32(x)), x);
    }
}

TEST(UtilIntrinsics, ByteSwap64KnownValues)
{
    EXPECT_EQ(byte_swap_64(0x0000000000000000ULL), 0x0000000000000000ULL);
    EXPECT_EQ(byte_swap_64(0x0000000000000001ULL), 0x0100000000000000ULL);
    EXPECT_EQ(byte_swap_64(0x0123456789abcdefULL), 0xefcdab8967452301ULL);
    EXPECT_EQ(byte_swap_64(0xffffffffffffffffULL), 0xffffffffffffffffULL);
}

TEST(UtilIntrinsics, ByteSwap64IsInvolution)
{
    // byte_swap_64() applied twice must be the identity, for any input.
    Splitmix64 rng(2024);
    for (int i = 0; i < 1000; ++i) {
        std::uint64_t const x = rng.get_uint64();
        EXPECT_EQ(byte_swap_64(byte_swap_64(x)), x);
    }
}
