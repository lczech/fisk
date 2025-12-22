#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// -----------------------------------------------------------------------------
//     do_not_optimize_u64()
// -----------------------------------------------------------------------------

inline void do_not_optimize_u64(std::uint64_t v)
{
    #if defined(__GNUC__) || defined(__clang__)
        asm volatile("" : : "r"(v) : "memory");
    #else
        volatile std::uint64_t sink = v;
        (void) sink;
    #endif
}

// -----------------------------------------------------------------------------
//     Result and Bench
// -----------------------------------------------------------------------------

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
}

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
    if( expected == 0 ) {
        std::cerr << "Sink is 0. That is technically possible, but rather unlikely, ";
        std::cerr << "and instead points to a wrong setup, where the sink is incrrectly computed.\n";
    }
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
//     bench() helper
// -----------------------------------------------------------------------------

// Inlined wrapper for the benchmarked function
template <class F>
inline Bench<std::decay_t<F>> bench(std::string_view name, F&& fn)
{
    return Bench<std::decay_t<F>>{name, std::forward<F>(fn)};
}

// -----------------------------------------------------------------------------
//     Microbench<Input>
// -----------------------------------------------------------------------------

template <class Input>
class Microbench
{
public:
    using InputType = Input;

    // Convenience alias to reuse the Bench type
    template <class F>
    using BenchT = Bench<F>;

    explicit Microbench(std::string title)
        : title_(std::move(title))
    {}

    // ------------------------------------------------------
    //     configuration (fluent)
    // ------------------------------------------------------

    Microbench& rounds(int r) {
        if (r <= 0) throw std::runtime_error("rounds must be > 0");
        rounds_ = r;
        return *this;
    }

    Microbench& repeats(int r) {
        if (r <= 0) throw std::runtime_error("repeats must be > 0");
        repeats_ = r;
        return *this;
    }

    // Default: 1 unit per input element (ns_per_op == ns per element)
    Microbench& units_per_element(double u) {
        if (u <= 0.0) throw std::runtime_error("units_per_element must be > 0");
        units_per_element_ = u;
        units_fn_ = nullptr; // disable custom units function
        return *this;
    }

    // Custom units function, e.g. `[](auto const& in){ return in.seq.size(); }`
    // so ns_per_op becomes ns per base.
    Microbench& units_fn(std::function<double(Input const&)> fn) {
        units_fn_ = std::move(fn);
        return *this;
    }

    Microbench& check_sinks(bool b) {
        check_sinks_ = b;
        return *this;
    }

    Microbench& print_results(bool b) {
        print_results_ = b;
        return *this;
    }

    // Convenience: create a Bench for this Microbench
    template <class F>
    static BenchT<std::decay_t<F>> make_bench(std::string_view name, F&& fn) {
        return bench(std::forward<std::string_view>(name), std::forward<F>(fn));
    }

    // ------------------------------------------------------
    //     running with fixed inputs
    // ------------------------------------------------------

    template <class... Benches>
    std::vector<Result> run(
        std::vector<Input> const& inputs,
        Benches const&... bs
    ) const {
        if (repeats_ == 1) {
            auto rs = run_once(inputs, bs...);
            print_if_needed(rs, /*best_of=*/false);
            return rs;
        }

        auto best = run_once(inputs, bs...);
        for (int k = 1; k < repeats_; ++k) {
            auto cur = run_once(inputs, bs...);
            if (cur.size() != best.size()) {
                throw std::runtime_error("Microbench::run: benchmark count changed across repeats");
            }
            for (std::size_t i = 0; i < cur.size(); ++i) {
                if (cur[i].ns_per_op < best[i].ns_per_op) {
                    best[i] = cur[i];
                }
            }
        }
        print_if_needed(best, /*best_of=*/true);
        std::cout << "\nSink: " << best.front().sink << "\n";
        return best;
    }

    // ------------------------------------------------------
    //     running with generated inputs per repeat
    // ------------------------------------------------------

    template <class MakeInputs, class... Benches>
    std::vector<Result> run(
        MakeInputs make_inputs,
        Benches const&... bs
    ) const {
        if (repeats_ <= 0) {
            throw std::runtime_error("Microbench::run: repeats must be > 0");
        }

        // First run: generate inputs and time once.
        auto inputs0 = make_inputs();
        auto best    = run_once(inputs0, bs...);

        // Further repeats: new inputs each time; keep best ns/op.
        for (int k = 1; k < repeats_; ++k) {
            auto inputs = make_inputs();
            auto cur    = run_once(inputs, bs...);

            if (cur.size() != best.size()) {
                throw std::runtime_error("Microbench::run: benchmark count changed");
            }
            for (std::size_t i = 0; i < cur.size(); ++i) {
                if (cur[i].ns_per_op < best[i].ns_per_op) {
                    best[i] = cur[i];
                }
            }
        }

        print_if_needed(best, /*best_of=*/true);
        std::cout << "\nSink: " << best.front().sink << "\n";
        return best;
    }

private:

    // ------------------------------------------------------
    //     Internal functions
    // ------------------------------------------------------

    // Sum of "work units" for a single pass over all inputs
    double compute_units_per_run(std::vector<Input> const& inputs) const {
        if (units_fn_) {
            double total = 0.0;
            for (auto const& in : inputs) {
                total += units_fn_(in);
            }
            if (total <= 0.0) {
                throw std::runtime_error("Microbench::compute_units_per_run: total work units must be > 0");
            }
            return total;
        }
        return units_per_element_ * static_cast<double>(inputs.size());
    }

    template <class F>
    Result run_one(
        std::vector<Input> const& inputs,
        double units_per_run,
        Bench<F> const& b
    ) const {
        std::uint64_t acc = 0;

        // warm-up
        for (auto const& in : inputs) {
            acc += static_cast<std::uint64_t>(b.fn(in));
        }
        do_not_optimize_u64(acc);

        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < rounds_; ++r) {
            for (auto const& in : inputs) {
                acc += static_cast<std::uint64_t>(b.fn(in));
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        do_not_optimize_u64(acc);

        double secs        = std::chrono::duration<double>(t1 - t0).count();
        double total_units = units_per_run * static_cast<double>(rounds_);
        double ns_per_unit = (secs * 1e9) / total_units;

        return Result{b.name, ns_per_unit, acc};
    }

    template <class... Benches>
    std::vector<Result> run_once(
        std::vector<Input> const& inputs,
        Benches const&... bs
    ) const {
        std::vector<Result> rs;
        rs.reserve(sizeof...(bs));

        double units_per_run = compute_units_per_run(inputs);

        (rs.push_back(run_one(inputs, units_per_run, bs)), ...);

        if (check_sinks_) {
            require_same_sinks(rs, /*fatal=*/true);
        }

        return rs;
    }

    void print_if_needed(std::vector<Result> const& rs, bool best_of) const {
        if (!print_results_) return;
        if (best_of) {
            std::cout << "\n=== " << title_ << " (best of " << repeats_ << ") ===\n";
        } else {
            std::cout << "\n=== " << title_ << " ===\n";
        }
        print(rs);
    }

    // ------------------------------------------------------
    //     Member variables
    // ------------------------------------------------------

    std::string title_;
    int rounds_  = 10;
    int repeats_ = 1;

    // Either a constant factor per element, or a custom function:
    double units_per_element_ = 1.0;
    std::function<double(Input const&)> units_fn_{};

    bool check_sinks_   = true;
    bool print_results_ = false;
};
