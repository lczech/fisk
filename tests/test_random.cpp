#include <cstdint>

#include "fisk/core/random.hpp"
#include "testing.hpp"

TEST(Random, Splitmix64DeterministicForSameSeed)
{
    Splitmix64 a(12345);
    Splitmix64 b(12345);
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(a.get_uint64(), b.get_uint64());
    }
}

TEST(Random, Splitmix64DifferentSeedsDiffer)
{
    Splitmix64 a(1);
    Splitmix64 b(2);
    EXPECT_NE(a.get_uint64(), b.get_uint64());
}

TEST(Random, Splitmix64SetSeedResetsSequence)
{
    Splitmix64 rng(42);
    std::uint64_t const first = rng.get_uint64();
    std::uint64_t const second = rng.get_uint64();

    rng.set_seed(42);
    EXPECT_EQ(rng.get_uint64(), first);
    EXPECT_EQ(rng.get_uint64(), second);
}

TEST(Random, Splitmix64GetDoubleInUnitRange)
{
    Splitmix64 rng(7);
    for (int i = 0; i < 1000; ++i) {
        double const d = rng.get_double();
        EXPECT_TRUE(d >= 0.0);
        EXPECT_TRUE(d < 1.0);
    }
}
