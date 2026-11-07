#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fisk/util/intrinsics.hpp"
#include "sink.hpp"

using namespace fisk;

// -----------------------------------------------------------------------------
//     Settings
// -----------------------------------------------------------------------------

/**
 * @brief The timing settings shared by all suites, set once from the command line in main().
 *
 * Each repeat times every variant of a case once, in turn, and the results are the statistics
 * over the repeats. Each timing is a sample of whole passes over the inputs, as many as reach
 * the sample time, counted per variant from its first warm-up. Every timing is preceded by a
 * warm-up of whole passes that lasts at least the warm-up time, as the other variants of the
 * repeat have run in between and replaced the caches and the branch history.
 */
struct MicrobenchSettings
{
    double sample_seconds = 0.010;
    double warmup_seconds = 0.001;
    std::size_t repeats = 5;
};

inline MicrobenchSettings microbench_settings;

// -----------------------------------------------------------------------------
//     Result and Bench
// -----------------------------------------------------------------------------

struct Result
{
    std::string_view name;

    // Statistics of the time per unit over the repeats. The minimum is the main value, as
    // noise only adds time.
    double ns_per_op;
    double ns_per_op_median;
    double ns_per_op_mean;

    // Passes over the inputs per sample, and number of samples.
    std::size_t rounds;
    std::size_t repeats;

    // The sum of the results of one pass over the inputs, see require_same_sinks().
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
            << " : " << std::right << std::setw(ns_w) << std::fixed << std::setprecision(prec)
            << r.ns_per_op
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
    os << "suite,case,benchmark,sink,ns_per_op,ns_per_op_median,ns_per_op_mean,rounds,repeats\n";
}

// sink_name identifies which Sink strategy (see sink.hpp) produced every row of this call: the
// same one for all of them, since it is chosen once per binary at compile time. Suites that don't
// use the Sink abstraction yet fall back to the default "sum", which already describes their
// plain accumulation.
inline void write_csv_rows(
    std::ostream& os,
    std::string_view suite,
    std::string_view case_label,
    std::vector<Result> const& results,
    std::string_view sink_name = "sum",
    int precision = 6
) {
    os << std::fixed << std::setprecision(precision);
    for (auto const& r : results) {
        os << suite << ","
           << case_label << ","
           << r.name << ","
           << sink_name << ","
           << r.ns_per_op << ","
           << r.ns_per_op_median << ","
           << r.ns_per_op_mean << ","
           << r.rounds << ","
           << r.repeats
           << "\n";
    }
}

// -----------------------------------------------------------------------------
//     require_same_sinks()
// -----------------------------------------------------------------------------

/**
 * Helper to check that all results have the same sink value of their hash function.
 * This ensures that all our implementations give the same result.
 */
inline void require_same_sinks(
    std::vector<Result> const& rs,
    bool fatal = true
) {
    if (rs.empty()) return;

    auto const expected = rs.front().sink;
    if( expected == 0 ) {
        std::cerr << "Accumulation sink is 0. That is possible for empty masks etc, ";
        std::cerr << "or points to a wrong setup, where the sink is incorrectly computed.\n";
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

/**
 * @brief Times the variants of one case, see MicrobenchSettings for how.
 */
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

    // Optional: compute a bench's sink once, after its timed rounds finish, instead of using the
    // accumulation of its per-call return values. For a "write" style bench (result left in some
    // shared state rather than returned per call), this keeps reducing that state out of the timed
    // region. Shared across every bench() passed to run(), called once per bench right after its
    // own timed loop -- safe as long as those benches run sequentially and don't overlap the state
    // this reads, which Microbench guarantees.
    Microbench& finalize(std::function<std::uint64_t()> fn) {
        finalize_ = std::move(fn);
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
        return run_repeats_([&]() -> std::vector<Input> const& { return inputs; }, bs...);
    }

    // ------------------------------------------------------
    //     running with generated inputs per repeat
    // ------------------------------------------------------

    template <class MakeInputs, class... Benches>
    std::vector<Result> run(
        MakeInputs make_inputs,
        Benches const&... bs
    ) const {
        std::vector<Input> inputs;
        return run_repeats_(
            [&]() -> std::vector<Input> const& { inputs = make_inputs(); return inputs; }, bs...
        );
    }

private:

    // ------------------------------------------------------
    //     Internal functions
    // ------------------------------------------------------

    // Per bench: the passes per sample (0 until counted in the first repeat), the time per unit
    // of each repeat, and the sink of the last repeat.
    struct State
    {
        std::size_t rounds = 0;
        std::vector<double> ns_per_unit;
        std::uint64_t sink = 0;
    };

    // Sum of "work units" for a single pass over all inputs
    double compute_units_per_run(std::vector<Input> const& inputs) const {
        if (units_fn_) {
            double total = 0.0;
            for (auto const& in : inputs) {
                total += units_fn_(in);
            }
            if (total <= 0.0) {
                throw std::runtime_error(
                    "Microbench::compute_units_per_run: total work units must be > 0"
                );
            }
            return total;
        }
        return units_per_element_ * static_cast<double>(inputs.size());
    }

    template <class NextInputs, class... Benches>
    std::vector<Result> run_repeats_(
        NextInputs next_inputs,
        Benches const&... bs
    ) const {
        auto const& settings = microbench_settings;
        if (settings.repeats == 0) {
            throw std::runtime_error("Microbench::run: repeats must be > 0");
        }

        std::array<State, sizeof...(Benches)> states{};
        for (std::size_t k = 0; k < settings.repeats; ++k) {
            auto const& inputs = next_inputs();
            double const units_per_run = compute_units_per_run(inputs);
            std::size_t i = 0;
            (run_one_(inputs, units_per_run, bs, states[i++]), ...);

            // The sinks are compared per repeat, as generated inputs differ between repeats.
            if (check_sinks_) {
                std::vector<Result> sinks;
                std::size_t j = 0;
                ((sinks.push_back(Result{bs.name, 0.0, 0.0, 0.0, 0, 0, states[j++].sink})), ...);
                require_same_sinks(sinks, /*fatal=*/true);
            }
        }

        std::vector<Result> rs;
        rs.reserve(sizeof...(Benches));
        std::size_t i = 0;
        (rs.push_back(make_result_(bs.name, states[i++])), ...);

        print_if_needed(rs);
        std::cout << "\nSink: " << rs.front().sink << "\n";
        return rs;
    }

    template <class F>
    void run_one_(
        std::vector<Input> const& inputs,
        double units_per_run,
        Bench<F> const& b,
        State& state
    ) const {
        using clock = std::chrono::steady_clock;
        auto const& settings = microbench_settings;
        std::uint64_t acc = 0;

        // Warm-up: whole passes until the warm-up time has passed. The sink is the sum of the
        // first pass, so that it does not depend on the number of passes of the variant.
        // Every pass runs on the same inputs, so without the memory barrier, the compiler could
        // compute an inlined kernel once and reuse its results for the other passes.
        std::size_t passes = 0;
        double warmup_secs = 0.0;
        auto const w0 = clock::now();
        do {
            clobber_memory(inputs.data());
            std::uint64_t pass_acc = 0;
            for (auto const& in : inputs) {
                pass_acc += static_cast<std::uint64_t>(b.fn(in));
            }
            if (passes == 0) {
                state.sink = pass_acc;
            }
            acc += pass_acc;
            ++passes;
            warmup_secs = std::chrono::duration<double>(clock::now() - w0).count();
        } while (warmup_secs < settings.warmup_seconds);
        do_not_optimize(acc);

        // The first repeat sets the passes per sample from the time per pass of its warm-up.
        if (state.rounds == 0) {
            double const pass_secs = warmup_secs / static_cast<double>(passes);
            state.rounds = static_cast<std::size_t>(
                std::max(1.0, std::ceil(settings.sample_seconds / pass_secs))
            );
        }

        auto const t0 = clock::now();
        for (std::size_t r = 0; r < state.rounds; ++r) {
            clobber_memory(inputs.data());
            for (auto const& in : inputs) {
                acc += static_cast<std::uint64_t>(b.fn(in));
            }
        }
        auto const t1 = clock::now();
        do_not_optimize(acc);

        double const secs        = std::chrono::duration<double>(t1 - t0).count();
        double const total_units = units_per_run * static_cast<double>(state.rounds);
        state.ns_per_unit.push_back((secs * 1e9) / total_units);
        if (finalize_) {
            state.sink = finalize_();
        }
    }

    static Result make_result_(std::string_view name, State const& state) {
        auto ns = state.ns_per_unit;
        std::sort(ns.begin(), ns.end());
        auto const n = ns.size();
        double const median = (n % 2 == 1) ? ns[n / 2] : (ns[n / 2 - 1] + ns[n / 2]) / 2.0;
        double const mean = std::accumulate(ns.begin(), ns.end(), 0.0) / static_cast<double>(n);
        return Result{name, ns.front(), median, mean, state.rounds, n, state.sink};
    }

    void print_if_needed(std::vector<Result> const& rs) const {
        if (!print_results_) return;
        std::cout << "\n=== " << title_ << " (min of " << microbench_settings.repeats << ") ===\n";
        print(rs);
    }

    // ------------------------------------------------------
    //     Member variables
    // ------------------------------------------------------

    std::string title_;

    // Either a constant factor per element, or a custom function:
    double units_per_element_ = 1.0;
    std::function<double(Input const&)> units_fn_{};

    // Only the Sum sink gives a value per pass that is comparable between variants; Barrier and
    // Write do not collect the results.
    bool check_sinks_   = FISK_BENCHMARK_SINK_MODE == FISK_BENCHMARK_SINK_SUM;
    bool print_results_ = false;

    std::function<std::uint64_t()> finalize_{};
};
