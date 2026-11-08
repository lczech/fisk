#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
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

template <bool Spaced, bool Vectors, typename Extractor>
void tune_scaling_variant(Extractor& extractor, std::vector<std::uint64_t>& sink_buffer)
{
    auto sink = make_sink(sink_buffer);
    extractor.tune(make_scaling_callback<Spaced, Vectors>(sink));
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
