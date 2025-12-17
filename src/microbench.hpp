#pragma once

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <ostream>

namespace microbench
{

inline void do_not_optimize_u64(std::uint64_t v)
{
    #if defined(__GNUC__) || defined(__clang__)
        asm volatile("" : : "r"(v) : "memory");
    #else
        volatile std::uint64_t sink = v;
        (void)sink;
    #endif
}

struct Result
{
    std::string_view name;
    double ns_per_op;
    std::uint64_t sink;
};

template <class F>
struct Bench
{
    std::string_view name;
    F fn;
};

// -----------------------------------------------------------------------------
//     print()
// -----------------------------------------------------------------------------

inline void print(std::vector<Result> const& rs)
{
    // name column width: longest name, minimum 10
    std::size_t name_w = 10;
    for (auto const& r : rs) name_w = std::max(name_w, r.name.size());

    // ns/op formatting:
    // - fixed precision (e.g. 3 decimals)
    // - width sized for up to 999.999 (plus a little breathing room)
    constexpr int prec = 3;
    constexpr int ns_w = 9; // "999.999" is 7 chars; 9 gives padding/alignment

    for (auto const& r : rs) {
        std::cout
            << "  " << std::left  << std::setw(static_cast<int>(name_w)) << r.name
            << " : " << std::right << std::setw(ns_w) << std::fixed << std::setprecision(prec) << r.ns_per_op
            << " ns/op"
            << "   sink=" << r.sink
            << "\n";
    }

    // Reset stream defaults that might surprise later printing
    std::cout.unsetf(std::ios::floatfield);
    std::cout << std::setprecision(6);
};

// -----------------------------------------------------------------------------
//     write_csv_...()
// -----------------------------------------------------------------------------

inline void write_csv_header(std::ostream& os)
{
    os << "suite,case,benchmark,ns_per_op\n";
}

inline void write_csv_rows(
    std::ostream& os,
    std::string_view suite,
    std::string_view case_label,
    std::vector<Result> const& results,
    int precision = 6
) {
    os << std::fixed << std::setprecision(precision);
    for (auto const& r : results) {
        os << suite << ","
           << case_label << ","
           << r.name << ","
           << r.ns_per_op
           << "\n";
    }
}

// -----------------------------------------------------------------------------
//     require_same_sinks()
// -----------------------------------------------------------------------------

inline void require_same_sinks(
    std::vector<Result> const& rs,
    bool fatal = true
) {
    if (rs.empty()) return;

    auto const expected = rs.front().sink;
    for (auto const& r : rs) {
        if (r.sink != expected) {
            std::cerr << "Sink mismatch!\n"
                      << "Expected sink (from '" << rs.front().name << "') = " << expected << "\n"
                      << "But '" << r.name << "' produced sink = " << r.sink << "\n";
            if (fatal) std::exit(3);
            return;
        }
    }
}

// -----------------------------------------------------------------------------
//     bench()
// -----------------------------------------------------------------------------

// Inlined wrapper for the benchmarked function
template <class F>
inline Bench<std::decay_t<F>> bench(std::string_view name, F&& fn)
{
    return Bench<std::decay_t<F>>{name, std::forward<F>(fn)};
}

// -----------------------------------------------------------------------------
//     run_one()
// -----------------------------------------------------------------------------

// Run a single benchmarked function, with no function call overhead,
// indirection, type erasure, or other confounding factors,
// as the compiler should be able to inline all of it.
template <class Input, class F>
Result run_one(
    std::vector<Input> const& inputs,
    int rounds,
    Bench<F> const& b
) {
    std::uint64_t acc = 0;

    // warm-up
    for (auto const& in : inputs) acc ^= static_cast<std::uint64_t>(b.fn(in));
    do_not_optimize_u64(acc);

    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < rounds; ++r) {
        for (auto const& in : inputs) {
            acc ^= static_cast<std::uint64_t>(b.fn(in));
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    do_not_optimize_u64(acc);

    double secs = std::chrono::duration<double>(t1 - t0).count();
    double ops  = static_cast<double>(inputs.size()) * static_cast<double>(rounds);
    return Result{b.name, (secs * 1e9) / ops, acc};
}

// -----------------------------------------------------------------------------
//     run_suite()
// -----------------------------------------------------------------------------

// Run a suite of functions to be benchmarked and compared to each other.
template <class Input, class... Benches>
std::vector<Result> run_suite(
    std::string_view title,
    std::vector<Input> const& inputs,
    int rounds,
    Benches const&... bs
) {
    (void) title;

    std::vector<Result> rs;
    rs.reserve(sizeof...(bs));

    (rs.push_back(run_one(inputs, rounds, bs)), ...);

    // std::cout << "\n=== " << title << " ===\n";
    // print(rs);
    require_same_sinks(rs);

    return rs;
}

// -----------------------------------------------------------------------------
//     run_suite_best_of()
// -----------------------------------------------------------------------------

// Run multiple repeats and report the best (min) ns/op per benchmark.
template <class MakeInputs, class... Benches>
std::vector<Result> run_suite_best_of(
    std::string_view title,
    MakeInputs make_inputs,
    int rounds,
    int repeats,
    Benches const&... bs
) {
    if (repeats <= 0) {
        throw std::runtime_error("run_suite_best_of_generated: repeats must be > 0");
    }

    // First run: generate inputs and time once.
    auto inputs0 = make_inputs();
    auto best = run_suite(title, inputs0, rounds, bs...);

    // Further repeats: new inputs each time; keep best ns/op.
    for (int k = 1; k < repeats; ++k) {
        auto inputs = make_inputs();
        auto cur = run_suite(title, inputs, rounds, bs...);

        if (cur.size() != best.size()) {
            throw std::runtime_error("run_suite_best_of_generated: benchmark count changed");
        }

        for (std::size_t i = 0; i < cur.size(); ++i) {
            if (cur[i].ns_per_op < best[i].ns_per_op) {
                best[i] = cur[i];
            }
        }
    }

    // std::cout << "\n=== " << title << " (best of " << repeats << ") ===\n";
    // print(best);
    return best;
}

} // namespace microbench
