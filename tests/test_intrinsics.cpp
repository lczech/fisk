#include <cstdint>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/random.hpp"
#include "testing.hpp"

TEST(Intrinsics, ByteSwap64KnownValues)
{
    EXPECT_EQ(byte_swap_64(0x0000000000000000ULL), 0x0000000000000000ULL);
    EXPECT_EQ(byte_swap_64(0x0000000000000001ULL), 0x0100000000000000ULL);
    EXPECT_EQ(byte_swap_64(0x0123456789abcdefULL), 0xefcdab8967452301ULL);
    EXPECT_EQ(byte_swap_64(0xffffffffffffffffULL), 0xffffffffffffffffULL);
}

TEST(Intrinsics, ByteSwap64IsInvolution)
{
    // byte_swap_64() applied twice must be the identity, for any input.
    Splitmix64 rng(2024);
    for (int i = 0; i < 1000; ++i) {
        std::uint64_t const x = rng.get_uint64();
        EXPECT_EQ(byte_swap_64(byte_swap_64(x)), x);
    }
}
