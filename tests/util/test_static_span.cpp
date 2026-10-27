#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "fisk/util/static_span.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Static Span Dispatch Tests
// =================================================================================================

// The values 1 to `n`, as input of the tests below.
static std::vector<int> iota_vector_(std::size_t n)
{
    std::vector<int> result(n);
    for (std::size_t i = 0; i < n; ++i) {
        result[i] = static_cast<int>(i + 1);
    }
    return result;
}

TEST(UtilStaticSpan, EverySizeUpToMax)
{
    // Each size in [1, MaxN] arrives as a span of that compile-time extent, over the elements of
    // the input, without a copy.
    constexpr std::size_t kMax = 8;
    for (std::size_t n = 1; n <= kMax; ++n) {
        auto const values = iota_vector_(n);
        std::size_t const extent = dispatch_static_span<kMax>(values, [&](auto span) {
            using Span = decltype(span);
            static_assert(Span::extent != std::dynamic_extent);
            static_assert(std::is_same_v<typename Span::element_type, int const>);
            EXPECT_EQ(span.size(), Span::extent);
            EXPECT_EQ(span.data(), values.data());
            return Span::extent;
        });
        EXPECT_EQ(extent, n);
    }
}

TEST(UtilStaticSpan, ThrowsOnEmpty)
{
    std::vector<int> const values;
    bool called = false;
    EXPECT_THROW(
        dispatch_static_span<4>(values, [&](auto) { called = true; }), std::invalid_argument
    );
    EXPECT_FALSE(called);
}

TEST(UtilStaticSpan, ThrowsAboveMax)
{
    auto const values = iota_vector_(5);
    bool called = false;
    EXPECT_THROW(
        dispatch_static_span<4>(values, [&](auto) { called = true; }), std::invalid_argument
    );
    EXPECT_FALSE(called);

    // The message names the size and the range of sizes.
    std::string message;
    try {
        dispatch_static_span<4>(values, [](auto) {});
    } catch (std::invalid_argument const& e) {
        message = e.what();
    }
    EXPECT_NE(message.find("size 5 not in [1, 4]"), std::string::npos);
}

TEST(UtilStaticSpan, MaxOfOne)
{
    auto const one = iota_vector_(1);
    EXPECT_EQ(dispatch_static_span<1>(one, [](auto span) { return span[0]; }), 1);

    auto const two = iota_vector_(2);
    EXPECT_THROW(dispatch_static_span<1>(two, [](auto) {}), std::invalid_argument);
}

TEST(UtilStaticSpan, MaxAboveDefaultMaskCount)
{
    // MaxN is a parameter of each call, and can exceed the mask default of 16.
    auto const values = iota_vector_(20);
    auto const sum = dispatch_static_span<24>(values, [](auto span) {
        EXPECT_EQ(decltype(span)::extent, 20u);
        int s = 0;
        for (auto v : span) {
            s += v;
        }
        return s;
    });
    EXPECT_EQ(sum, 210);
}

TEST(UtilStaticSpan, ContiguousRanges)
{
    // Every range with std::data() and std::size() works, with const elements in the span.
    auto const check = [](auto const& range) {
        return dispatch_static_span<4>(range, [&](auto span) {
            static_assert(std::is_const_v<typename decltype(span)::element_type>);
            EXPECT_EQ(span.data(), std::data(range));
            return span[0] + span[span.size() - 1] + static_cast<int>(span.size());
        });
    };

    std::vector<int> const vector{1, 2, 3};
    std::array<int, 3> const array{1, 2, 3};
    int c_array[3] = {1, 2, 3};
    std::vector<int> mutable_vector{1, 2, 3};
    std::span<int> const dynamic_span(mutable_vector);
    std::span<int const, 3> const static_span(vector);

    EXPECT_EQ(check(vector), 7);
    EXPECT_EQ(check(array), 7);
    EXPECT_EQ(check(c_array), 7);
    EXPECT_EQ(check(dynamic_span), 7);
    EXPECT_EQ(check(static_span), 7);
}

TEST(UtilStaticSpan, VoidResult)
{
    auto const values = iota_vector_(3);
    std::size_t seen = 0;
    dispatch_static_span<4>(values, [&](auto span) { seen = span.size(); });
    static_assert(std::is_void_v<decltype(dispatch_static_span<4>(values, [](auto) {}))>);
    EXPECT_EQ(seen, 3u);
}

TEST(UtilStaticSpan, ReferenceResult)
{
    // A reference that f returns comes back as that reference, to the element of the input.
    auto const values = iota_vector_(3);
    int const& last = dispatch_static_span<4>(values, [](auto span) -> int const& {
        return span[span.size() - 1];
    });
    EXPECT_EQ(&last, &values[2]);
}

TEST(UtilStaticSpan, CalledOnce)
{
    auto const values = iota_vector_(3);
    int calls = 0;
    dispatch_static_span<4>(values, [&](auto) { ++calls; });
    EXPECT_EQ(calls, 1);
}

// A functor with state, for the test below.
struct StaticSpanCounter_
{
    std::size_t total = 0;

    template <typename Span>
    void operator()(Span span)
    {
        total += span.size();
    }
};

TEST(UtilStaticSpan, StatefulFunctor)
{
    // f is called as the object passed in, so that a functor keeps its state.
    StaticSpanCounter_ counter;
    auto const values = iota_vector_(3);
    dispatch_static_span<4>(values, counter);
    dispatch_static_span<4>(values, counter);
    EXPECT_EQ(counter.total, 6u);
}

TEST(UtilStaticSpan, ExceptionFromFunction)
{
    auto const values = iota_vector_(2);
    EXPECT_THROW(
        dispatch_static_span<4>(values, [](auto) -> int { throw std::runtime_error("f"); }),
        std::runtime_error
    );
}

TEST(UtilStaticSpan, MoveOnlyElements)
{
    // The elements are neither copied nor moved, so they need not be copyable.
    std::vector<std::unique_ptr<int>> values;
    values.push_back(std::make_unique<int>(5));
    values.push_back(std::make_unique<int>(7));
    auto const sum = dispatch_static_span<4>(values, [](auto span) {
        return *span[0] + *span[1];
    });
    EXPECT_EQ(sum, 12);
}
