#include <cstdint>
#include <string>
#include <string_view>

#include "for_each_scaling/bench.hpp"
#include "for_each_scaling/variant.hpp"

std::uint64_t run_var_kmer_checked(std::string_view seq, SinkBuffer& sink_buffer)
{
    return run_scaling_kmer</*Vectors=*/false, fisk::InputCheck::kChecked>(seq, sink_buffer);
}

std::string name_var_kmer_checked()
{
    return scaling_kmer_name</*Vectors=*/false, fisk::InputCheck::kChecked>();
}
