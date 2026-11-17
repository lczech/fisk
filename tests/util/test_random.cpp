#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "fisk/util/random.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Random Tests
// =================================================================================================

TEST(UtilRandom, Splitmix64DeterministicForSameSeed)
{
    Splitmix64 a(12345);
    Splitmix64 b(12345);
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(a.get_uint64(), b.get_uint64());
    }
}

TEST(UtilRandom, Splitmix64DifferentSeedsDiffer)
{
    Splitmix64 a(1);
    Splitmix64 b(2);
    EXPECT_NE(a.get_uint64(), b.get_uint64());
}

TEST(UtilRandom, Splitmix64SetSeedResetsSequence)
{
    Splitmix64 rng(42);
    std::uint64_t const first = rng.get_uint64();
    std::uint64_t const second = rng.get_uint64();

    rng.set_seed(42);
    EXPECT_EQ(rng.get_uint64(), first);
    EXPECT_EQ(rng.get_uint64(), second);
}

TEST(UtilRandom, Splitmix64GetDoubleInUnitRange)
{
    Splitmix64 rng(7);
    for (int i = 0; i < 1000; ++i) {
        double const d = rng.get_double();
        EXPECT_TRUE(d >= 0.0);
        EXPECT_TRUE(d < 1.0);
    }
}

// random_acgt(): the length, only ACGT without a probability of N, the same string for the same
// seed and another for another seed, N at about their probability, all N at 1, and the start of the
// string at a cache line.
TEST(UtilRandom, RandomAcgt)
{
    auto const seq = fisk::random_acgt(10000);
    EXPECT_EQ(seq.size(), std::size_t{10000});
    for (char c : seq) {
        EXPECT_TRUE(c == 'A' || c == 'C' || c == 'G' || c == 'T');
    }
    EXPECT_TRUE(seq == fisk::random_acgt(10000));
    EXPECT_TRUE(seq != fisk::random_acgt(10000, 0.0, 43));
    EXPECT_EQ(fisk::random_acgt().size(), std::size_t{1} << 20);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(seq.data()) % 64, std::uintptr_t{0});

    auto const with_n = fisk::random_acgt(100000, 0.25, 7);
    std::size_t n = 0;
    for (char c : with_n) {
        n += c == 'N';
    }
    EXPECT_TRUE(n > 24000 && n < 26000);
    for (char c : fisk::random_acgt(100, 1.0)) {
        EXPECT_EQ(c, 'N');
    }
    bool thrown = false;
    try {
        (void) fisk::random_acgt(10, 1.5);
    } catch (std::invalid_argument const&) {
        thrown = true;
    }
    EXPECT_TRUE(thrown);
}
