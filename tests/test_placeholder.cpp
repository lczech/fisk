// Smoke test for the test framework and CMake/CTest wiring themselves; not a test of fisk. Remove
// once real test files (e.g. test_seq_pack.cpp) exercise the library and prove the harness works.

#include "testing.hpp"

TEST(Placeholder, Sanity)
{
    ASSERT_EQ(1, 1);
    EXPECT_TRUE(true);
}
