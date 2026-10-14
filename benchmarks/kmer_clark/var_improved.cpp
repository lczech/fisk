#include <cstdint>
#include <string>
#include <vector>

#include "external/clark.hpp"
#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/kmer_spaced/kmer_spaced.hpp"
#include "kmer_clark/bench.hpp"

namespace {

std::vector<fisk::BitExtractMask> make_masks()
{
    std::vector<std::string> const mask_strings = {
        "1111011101110010111001011011111",
        "1111101011100101101110011011111",
        "1111101001110101101100111011111"
    };
    std::vector<fisk::BitExtractMask> masks;
    for (auto const& mask : mask_strings) {
        masks.push_back(fisk::make_spaced_kmer_mask(mask));
    }
    return masks;
}

std::vector<fisk::BitExtractMask> const masks = make_masks();

} // anonymous namespace

std::uint64_t run_var_improved(std::string const& seq)
{
    return clark::clark_improved(seq, masks);
}
