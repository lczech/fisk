#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/core/random.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/packed.hpp"
#include "fisk/seq_pack/seq_pack.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers
// =================================================================================================

// The oracle (oracle.hpp) decodes directly from the ASCII input, independent of the packed
// extractors under test; the packing step itself is checked in test_seq_pack.cpp.

// Sequence lengths 0..512, each with random per-position content.
static std::vector<std::string> const& test_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        Splitmix64 rng(5005);
        for (std::size_t len = 0; len <= 512; ++len) {
            out.push_back(random_sequence(rng, len));
        }
        return out;
    }();
    return seqs;
}

// Checks one packed extractor, called as `extract(packed, k, callback)`, under the conventions of
// E and L: the complete ordered output, positions and values, in both callback forms, for every k
// and every test sequence, plus its invalid-k contract.
template <Encoding E, Layout L, typename Extract>
static void check_packed_variant(Extract const& extract)
{
    WordEncoderButterfly<E, L> const encoder;
    sweep_packed_extractors(
        encoder, test_sequences(), 32,
        [&](std::string const&, auto const& packed, std::size_t k, auto const& expected) {
            check_kmer_callbacks<E, L>(expected, k, [&](auto const& callback) {
                extract(packed, k, callback);
            });
        }
    );
    check_invalid_k_throws(32, [&](std::string const& seq, std::size_t k) {
        extract(pack_sequence(seq, encoder), k, [](Kmer<E, L>) {});
    });
}

// The extractors under test, each wrapped once so that it can be handed to the checks above under
// every convention.
static constexpr auto packed_rolling = [](auto const& seq, std::size_t k, auto const& func) {
    for_each_kmer_packed_rolling(seq, k, func);
};
static constexpr auto packed_aligned = [](auto const& seq, std::size_t k, auto const& func) {
    for_each_kmer_packed_aligned(seq, k, func);
};

// =================================================================================================
//     for_each_kmer_packed_rolling()
// =================================================================================================

TEST(KmerExtractPacked, RollingMsb)
{
    check_packed_variant<Encoding::kACGT, Layout::kMSB>(packed_rolling);
}

TEST(KmerExtractPacked, RollingLsb)
{
    check_packed_variant<Encoding::kACGT, Layout::kLSB>(packed_rolling);
}

TEST(KmerExtractPacked, RollingActgMsb)
{
    check_packed_variant<Encoding::kACTG, Layout::kMSB>(packed_rolling);
}

TEST(KmerExtractPacked, RollingActgLsb)
{
    check_packed_variant<Encoding::kACTG, Layout::kLSB>(packed_rolling);
}

// =================================================================================================
//     for_each_kmer_packed_aligned()
// =================================================================================================

TEST(KmerExtractPacked, AlignedMsb)
{
    check_packed_variant<Encoding::kACGT, Layout::kMSB>(packed_aligned);
}

TEST(KmerExtractPacked, AlignedLsb)
{
    check_packed_variant<Encoding::kACGT, Layout::kLSB>(packed_aligned);
}

TEST(KmerExtractPacked, AlignedActgMsb)
{
    check_packed_variant<Encoding::kACTG, Layout::kMSB>(packed_aligned);
}

TEST(KmerExtractPacked, AlignedActgLsb)
{
    check_packed_variant<Encoding::kACTG, Layout::kLSB>(packed_aligned);
}
