#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

#include "fisk/core/cast.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "sink.hpp"

// =================================================================================================
//     Variants of the Extractor Scaling
// =================================================================================================

// The tuning and the run of one variant, for its var_*.cpp, which instantiates them in its own
// translation unit. Both use the same callback, so that tune() chooses the fastest algorithm for
// it: the k-mers, or spaced k-mers, into the sink, one at a time, or, with Vectors, also the
// vectors of the SIMD algorithms, with their valid lanes only, see consume_valid_lanes() (sink.hpp),
// as the vector variants of the other suites. Each function constructs its own local Sink (see
// sink.hpp), so that Sum's accumulation can auto-vectorize with the extraction.

template <bool Spaced, bool Vectors, typename Sink>
auto make_scaling_callback(Sink& sink)
{
    if constexpr (Spaced) {
        auto single = [&sink](std::size_t, std::size_t, auto kmer) FISK_ALWAYS_INLINE_LAMBDA {
            sink.consume(fisk::kmer_value(kmer));
        };
        if constexpr (Vectors) {
            return fisk::callback_overload{
                single,
                [&sink](
                    std::size_t, std::size_t, auto kmers, auto valid
                ) FISK_ALWAYS_INLINE_LAMBDA {
                    consume_valid_lanes(sink, kmers, valid);
                }
            };
        } else {
            return single;
        }
    } else {
        auto single = [&sink](std::size_t, auto kmer) FISK_ALWAYS_INLINE_LAMBDA {
            sink.consume(fisk::kmer_value(kmer));
        };
        if constexpr (Vectors) {
            return fisk::callback_overload{
                single,
                [&sink](
                    std::size_t, auto kmers, auto valid
                ) FISK_ALWAYS_INLINE_LAMBDA {
                    consume_valid_lanes(sink, kmers, valid);
                }
            };
        } else {
            return single;
        }
    }
}

// The callback of tune(): it owns its sink, and calls the callback of make_scaling_callback() on
// it. tune() moves a callback that is passed as a temporary into a local of its own, see
// KmerExtractor::tune(), so that the sink is local to the timing there, as it is to the extraction
// in run_scaling_variant(), and the candidates are timed as they run there.
template <bool Spaced, bool Vectors>
struct ScalingTuneCallback
{
    Sink sink;

    template <typename... Args>
        requires std::invocable<
            decltype(make_scaling_callback<Spaced, Vectors>(std::declval<Sink&>()))&, Args...
        >
    FISK_ALWAYS_INLINE void operator()(Args... args)
    {
        make_scaling_callback<Spaced, Vectors>(sink)(args...);
    }
};

template <bool Spaced, bool Vectors, typename Extractor>
void tune_scaling_variant(Extractor& extractor, std::vector<std::uint64_t>& sink_buffer)
{
    extractor.tune(ScalingTuneCallback<Spaced, Vectors>{make_sink(sink_buffer)});
}

template <bool Spaced, bool Vectors, typename Extractor>
std::uint64_t run_scaling_variant(
    Extractor const& extractor,
    std::string_view seq,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    extractor.for_each(seq, make_scaling_callback<Spaced, Vectors>(sink));
    return sink.finalize();
}
