#pragma once

// Minimal, dependency-free unit test framework, deliberately named and shaped after GoogleTest
// (TEST / EXPECT_* / ASSERT_* / RUN_ALL_TESTS) so that test files written against this header
// would need little to no changes if the project ever adopts real gtest instead. It intentionally
// does not replicate fixtures (TEST_F), typed/parameterized tests, or gtest's full value-printing
// machinery -- only what plain, fixture-less value comparisons need.
//
// Usage:
//
//   TEST(SuiteName, TestName)
//   {
//       EXPECT_EQ(2 + 2, 4);
//       ASSERT_TRUE(some_condition);
//   }
//
// Test cases self-register on construction, so no manual test list needs to be maintained; a
// single call to RUN_ALL_TESTS() (typically from a dedicated tests/main.cpp) runs everything that
// got linked into the binary. As in gtest, `SuiteName.TestName` must be unique across the whole
// binary -- reusing a pair produces a duplicate-symbol build error, not a silent overwrite.
//
// EXPECT_* records a failure and lets the test continue; ASSERT_* records a failure and returns
// from the test function immediately. Because that `return` targets the enclosing test function,
// ASSERT_* (like in real gtest) must only be used directly inside a TEST(...) body, not inside a
// helper function it calls.

#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fisk_testing {

namespace detail {

// -------------------------------------------------------------------------------------------
//     Registration
// -------------------------------------------------------------------------------------------

struct TestCase
{
    char const* suite_name;
    char const* test_name;
    void (*run)();
};

// Function-local static: safe to call from static initializers in any translation unit,
// regardless of static-initialization order across those units.
inline std::vector<TestCase>& registry()
{
    static std::vector<TestCase> tests;
    return tests;
}

struct Registrar
{
    Registrar(char const* suite_name, char const* test_name, void (*run)())
    {
        registry().push_back(TestCase{suite_name, test_name, run});
    }
};

// -------------------------------------------------------------------------------------------
//     Failure Reporting
// -------------------------------------------------------------------------------------------

inline bool& current_test_failed()
{
    static bool failed = false;
    return failed;
}

inline void record_failure(char const* file, int line, std::string const& message)
{
    std::cerr << file << ":" << line << ": Failure\n" << message << "\n\n";
    current_test_failed() = true;
}

// Streams `value` via operator<< when available, so failure messages show it; falls back to a
// placeholder for types that do not support it, rather than failing to compile.
template <typename T>
std::string to_string_or_placeholder(T const& value)
{
    if constexpr (requires (std::ostream& os) { os << value; }) {
        std::ostringstream oss;
        oss << value;
        return oss.str();
    } else {
        return "<value of unprintable type>";
    }
}

inline std::string format_binary_failure(
    char const* lhs_expr, char const* rhs_expr, char const* op,
    std::string const& lhs_str, std::string const& rhs_str
) {
    std::ostringstream oss;
    oss << "Expected: (" << lhs_expr << ") " << op << " (" << rhs_expr << ")\n"
        << "  Actual: " << lhs_str << " vs " << rhs_str;
    return oss.str();
}

inline std::string format_bool_failure(char const* expr, bool expected)
{
    std::ostringstream oss;
    oss << "Value of: " << expr << "\n"
        << "  Actual: " << (expected ? "false" : "true") << "\n"
        << "Expected: " << (expected ? "true" : "false");
    return oss.str();
}

} // namespace detail

// -------------------------------------------------------------------------------------------
//     Runner
// -------------------------------------------------------------------------------------------

/**
 * @brief Run every TEST(...) case linked into the binary, printing gtest-style progress and a
 * summary. Returns 0 if all tests passed, 1 otherwise -- suitable to return directly from main().
 */
inline int run_all_tests()
{
    auto const& tests = detail::registry();
    std::cout << "[==========] Running " << tests.size()
              << " test" << (tests.size() == 1 ? "" : "s") << ".\n";

    std::size_t passed = 0;
    std::vector<std::string> failed_names;

    for (auto const& t : tests) {
        std::cout << "[ RUN      ] " << t.suite_name << "." << t.test_name << "\n";
        detail::current_test_failed() = false;
        t.run();
        if (detail::current_test_failed()) {
            std::cout << "[  FAILED  ] " << t.suite_name << "." << t.test_name << "\n";
            failed_names.push_back(std::string(t.suite_name) + "." + t.test_name);
        } else {
            std::cout << "[       OK ] " << t.suite_name << "." << t.test_name << "\n";
            ++passed;
        }
    }

    std::cout << "[==========] " << tests.size()
              << " test" << (tests.size() == 1 ? "" : "s") << " ran.\n";
    std::cout << "[  PASSED  ] " << passed << " test" << (passed == 1 ? "" : "s") << ".\n";
    if (!failed_names.empty()) {
        std::cout << "[  FAILED  ] " << failed_names.size()
                  << " test" << (failed_names.size() == 1 ? "" : "s") << ", listed below:\n";
        for (auto const& name : failed_names) {
            std::cout << "[  FAILED  ] " << name << "\n";
        }
    }

    return failed_names.empty() ? 0 : 1;
}

} // namespace fisk_testing

// =================================================================================================
//     Macros
// =================================================================================================

#define RUN_ALL_TESTS() ::fisk_testing::run_all_tests()

#define TEST(suite_name, test_name) \
    static void suite_name##_##test_name##_Test(); \
    namespace { \
        ::fisk_testing::detail::Registrar const suite_name##_##test_name##_Registrar( \
            #suite_name, #test_name, &suite_name##_##test_name##_Test \
        ); \
    } \
    static void suite_name##_##test_name##_Test()

// -----------------------------------------------------------------------------
//     EXPECT_EQ / ASSERT_EQ / EXPECT_NE / ASSERT_NE
// -----------------------------------------------------------------------------

#define FISK_TESTING_EQ_(lhs, rhs, on_fail) \
    do { \
        auto const& fisk_testing_lhs_ = (lhs); \
        auto const& fisk_testing_rhs_ = (rhs); \
        if (!(fisk_testing_lhs_ == fisk_testing_rhs_)) { \
            ::fisk_testing::detail::record_failure(__FILE__, __LINE__, \
                ::fisk_testing::detail::format_binary_failure( \
                    #lhs, #rhs, "==", \
                    ::fisk_testing::detail::to_string_or_placeholder(fisk_testing_lhs_), \
                    ::fisk_testing::detail::to_string_or_placeholder(fisk_testing_rhs_) \
                ) \
            ); \
            on_fail; \
        } \
    } while (0)

#define FISK_TESTING_NE_(lhs, rhs, on_fail) \
    do { \
        auto const& fisk_testing_lhs_ = (lhs); \
        auto const& fisk_testing_rhs_ = (rhs); \
        if (fisk_testing_lhs_ == fisk_testing_rhs_) { \
            ::fisk_testing::detail::record_failure(__FILE__, __LINE__, \
                ::fisk_testing::detail::format_binary_failure( \
                    #lhs, #rhs, "!=", \
                    ::fisk_testing::detail::to_string_or_placeholder(fisk_testing_lhs_), \
                    ::fisk_testing::detail::to_string_or_placeholder(fisk_testing_rhs_) \
                ) \
            ); \
            on_fail; \
        } \
    } while (0)

#define EXPECT_EQ(lhs, rhs) FISK_TESTING_EQ_(lhs, rhs, (void) 0)
#define ASSERT_EQ(lhs, rhs) FISK_TESTING_EQ_(lhs, rhs, return)
#define EXPECT_NE(lhs, rhs) FISK_TESTING_NE_(lhs, rhs, (void) 0)
#define ASSERT_NE(lhs, rhs) FISK_TESTING_NE_(lhs, rhs, return)

// -----------------------------------------------------------------------------
//     EXPECT_TRUE / ASSERT_TRUE / EXPECT_FALSE / ASSERT_FALSE
// -----------------------------------------------------------------------------

#define FISK_TESTING_BOOL_(cond, expected, on_fail) \
    do { \
        bool const fisk_testing_cond_ = static_cast<bool>(cond); \
        if (fisk_testing_cond_ != (expected)) { \
            ::fisk_testing::detail::record_failure(__FILE__, __LINE__, \
                ::fisk_testing::detail::format_bool_failure(#cond, (expected)) \
            ); \
            on_fail; \
        } \
    } while (0)

#define EXPECT_TRUE(cond)  FISK_TESTING_BOOL_(cond, true, (void) 0)
#define ASSERT_TRUE(cond)  FISK_TESTING_BOOL_(cond, true, return)
#define EXPECT_FALSE(cond) FISK_TESTING_BOOL_(cond, false, (void) 0)
#define ASSERT_FALSE(cond) FISK_TESTING_BOOL_(cond, false, return)
