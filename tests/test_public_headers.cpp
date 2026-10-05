// This translation unit intentionally includes only the documented public umbrella header.
// It catches missing direct includes and include-order dependencies in the user-facing entry point.
#include "fisk/fisk.hpp"

#include "testing.hpp"

TEST(PublicHeaders, UmbrellaHeaderCompiles)
{
    // Compilation is the contract under test; no runtime behavior is needed here.
}
