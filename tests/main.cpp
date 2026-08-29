// Entry point for the fisk_tests binary. Kept separate from the individual test_*.cpp files so
// that adding a new test file only ever means adding TEST(...) cases, never a second main().

#include "testing.hpp"

int main()
{
    return RUN_ALL_TESTS();
}
