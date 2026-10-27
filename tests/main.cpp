// Entry point for the fisk_tests binary. Kept separate from the individual test_*.cpp files so
// that adding a new test file only ever means adding TEST(...) cases, never a second main().

#include <iostream>

#include "fisk/core/cpu_runtime.hpp"
#include "testing.hpp"

int main()
{
    // Which code paths the tests cover depends on the instruction sets compiled in and on the CPU
    // that runs them, so the log states both.
    fisk::info_print_system(std::cout);
    std::cout << "\n";

    return RUN_ALL_TESTS();
}
