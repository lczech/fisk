#include <cstdint>
#include <string_view>

#include "for_each_scaling/bench.hpp"
#include "for_each_scaling/variant.hpp"

std::uint64_t run_var_kmer_unchecked_vectors(std::string_view seq, SinkBuffer& sink_buffer)
{
    return run_scaling_kmer</*Vectors=*/true, fisk::InputCheck::kUnchecked>(seq, sink_buffer);
}
