#include <cstdint>
#include <string>
#include <string_view>

#include "for_each_scaling/bench.hpp"
#include "for_each_scaling/variant.hpp"

std::uint64_t run_var_spaced_kmer_checked_vectors(
    std::string_view seq, ScalingMaskForms const& mask_forms, SinkBuffer& sink_buffer
) {
    return run_scaling_spaced_kmer</*Vectors=*/true, fisk::InputCheck::kChecked>(
        seq, mask_forms, sink_buffer
    );
}

std::string name_var_spaced_kmer_checked_vectors(ScalingMaskForms const& mask_forms)
{
    return scaling_spaced_kmer_name</*Vectors=*/true, fisk::InputCheck::kChecked>(mask_forms);
}
