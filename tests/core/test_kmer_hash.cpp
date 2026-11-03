#include <cstddef>
#include <functional>
#include <unordered_set>

#include "fisk/core/cast.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_hash.hpp"
#include "fisk/core/types.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     K-mers
// =================================================================================================

// -------------------------------------------------------------------------------------------------
//     Hashing
// -------------------------------------------------------------------------------------------------

// Both hashers must be usable, agree with equality, and the default must actually mix.
TEST(CoreKmerHash, Hashing)
{
    std::size_t const width = 12;
    auto const seqs = random_seqs(width, 64, 321321);

    std::unordered_set<KmerAcgtMsb> mixed;
    std::unordered_set<KmerAcgtMsb, KmerHashIdentity> identity;
    for (auto const& seq : seqs) {
        auto const kmer = oracle_kmer<KmerAcgtMsb>(seq);
        mixed.insert(kmer);
        identity.insert(kmer);

        // Equal k-mers hash equally, under both hashers.
        EXPECT_EQ(std::hash<KmerAcgtMsb>{}(kmer), std::hash<KmerAcgtMsb>{}(kmer));
        EXPECT_EQ(KmerHashIdentity{}(kmer), static_cast<std::size_t>(kmer_value(kmer)));
    }
    EXPECT_EQ(mixed.size(), identity.size());

    // The default hasher is the mixing one, not the identity.
    auto const probe = oracle_kmer<KmerAcgtMsb>(seqs.front());
    EXPECT_EQ(std::hash<KmerAcgtMsb>{}(probe), KmerHashMix{}(probe));
    EXPECT_NE(std::hash<KmerAcgtMsb>{}(probe), KmerHashIdentity{}(probe));
}

// =================================================================================================
//     Spaced K-mers
// =================================================================================================

// -------------------------------------------------------------------------------------------------
//     Hashing
// -------------------------------------------------------------------------------------------------

TEST(CoreKmerHash, SpacedHashing)
{
    auto const probe = spaced_kmer_cast<Encoding::kACGT, Layout::kMSB>(0x1B, 3);
    EXPECT_EQ(std::hash<SpacedKmerAcgtMsb>{}(probe), KmerHashMix{}(probe));
    EXPECT_NE(std::hash<SpacedKmerAcgtMsb>{}(probe), KmerHashIdentity{}(probe));
}

TEST(CoreKmerHash, SpacedUnorderedContainers)
{
    std::size_t const width = 12;
    auto const seqs = random_seqs(width, 64, 135790);

    std::unordered_set<SpacedKmerAcgtMsb> spaced_set;
    std::unordered_set<KmerAcgtMsb> contiguous_set;
    for (auto const& seq : seqs) {
        auto const contiguous = oracle_kmer<KmerAcgtMsb>(seq);
        auto const spaced = spaced_kmer_cast<Encoding::kACGT, Layout::kMSB>(
            kmer_value(contiguous), width
        );
        spaced_set.insert(spaced);
        contiguous_set.insert(contiguous);
        EXPECT_EQ(spaced_set.count(spaced), std::size_t{1});
    }
    EXPECT_EQ(spaced_set.size(), contiguous_set.size());
}
