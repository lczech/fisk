#include <cstdint>
#include <string>
#include <vector>

#include "external/clark.hpp"
#include "kmer_clark/bench.hpp"

namespace {

// A name suffices here: the actual mask bit patterns are hard-coded in clark, keyed by name.
std::vector<std::string> const mask_names = { "T295", "T38570", "T58570" };

} // anonymous namespace

std::uint64_t run_var_original(std::string const& seq)
{
    return clark_getObjectsDataComputeFull(seq, mask_names);
}
