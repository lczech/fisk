#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "fisk/core/cast.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/extractor/for_each.hpp"
#include "for_each_scaling/bench.hpp"
#include "sink.hpp"

// =================================================================================================
//     Variants of the For Each Scaling
// =================================================================================================

// The run of one variant, and the name of its algorithm, for its var_*.cpp, which instantiates
// them in its own translation unit. The callback hands the k-mers, or spaced k-mers, to the sink,
// one at a time, or, with Vectors, also the vectors of the SIMD algorithms, with their valid lanes
// only, see consume_valid_lanes() (sink.hpp), as the vector variants of the other suites. The run
// constructs its own local Sink (see sink.hpp), so that Sum's accumulation can auto-vectorize with
// the extraction.

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

template <bool Vectors, fisk::InputCheck C>
std::uint64_t run_scaling_kmer(std::string_view seq, SinkBuffer& sink_buffer)
{
    auto sink = make_sink(sink_buffer);
    fisk::for_each_kmer<C, fisk::Encoding::kACGT, fisk::Layout::kLSB>(
        seq, kScalingK, make_scaling_callback</*Spaced=*/false, Vectors>(sink)
    );
    return sink.finalize();
}

template <bool Vectors, fisk::InputCheck C>
std::uint64_t run_scaling_spaced_kmer(
    std::string_view seq,
    ScalingMaskForms const& mask_forms,
    SinkBuffer& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    fisk::for_each_spaced_kmer<C, fisk::Encoding::kACGT, fisk::Layout::kLSB>(
        seq, mask_forms, make_scaling_callback</*Spaced=*/true, Vectors>(sink)
    );
    return sink.finalize();
}

template <bool Vectors, fisk::InputCheck C>
std::string scaling_kmer_name()
{
    SinkBuffer sink_buffer = make_sink_buffer();
    auto sink = make_sink(sink_buffer);
    return fisk::for_each_kmer_algorithm_name<C, fisk::Encoding::kACGT, fisk::Layout::kLSB>(
        std::string_view{}, kScalingK, make_scaling_callback</*Spaced=*/false, Vectors>(sink)
    );
}

template <bool Vectors, fisk::InputCheck C>
std::string scaling_spaced_kmer_name(ScalingMaskForms const& mask_forms)
{
    SinkBuffer sink_buffer = make_sink_buffer();
    auto sink = make_sink(sink_buffer);
    return fisk::for_each_spaced_kmer_algorithm_name<C, fisk::Encoding::kACGT, fisk::Layout::kLSB>(
        std::string_view{}, mask_forms, make_scaling_callback</*Spaced=*/true, Vectors>(sink)
    );
}
