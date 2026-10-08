#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_spaced.hpp"
#include "fisk/core/kmer_wide.hpp"
#include "fisk/core/random.hpp"

namespace fisk {

// =================================================================================================
//     K-mer Hashing
// =================================================================================================

/**
 * @brief Hash a k-mer by its raw word, without mixing.
 *
 * Free, but note that k-mers adjacent in a sequence differ only by a shift, so their raw words
 * cluster heavily. Use this only when the k-mers are already well distributed, or when the
 * container does its own mixing. Otherwise prefer KmerHashMix, which is the default.
 *
 * A WideKmer hashes as its lowest word only, i.e. its last 32 bases under kMSB or its first 32
 * under kLSB, so the same caveat applies all the more.
 */
struct KmerHashIdentity
{
    template <KmerType K>
    [[nodiscard]] std::size_t operator()(K kmer) const noexcept
    {
        return static_cast<std::size_t>(kmer.value);
    }

    template <WideKmerType K>
    [[nodiscard]] std::size_t operator()(K const& kmer) const noexcept
    {
        return static_cast<std::size_t>(kmer.words[0]);
    }
};

/**
 * @brief Hash a k-mer through the splitmix64 finalizer.
 *
 * The same mix64_() that Splitmix64::get_uint64() applies to its counter (see core/random.hpp),
 * applied to the k-mer word instead. A handful of cycles, against the cache miss that the hash
 * table lookup around it costs anyway, in exchange for full avalanche.
 *
 * A WideKmer chains its words through mix64_(), lowest first, so that every word affects the
 * result. A single word thus hashes as the Kmer of the same value.
 *
 * This is what std::hash<Kmer> uses. To opt out, name KmerHashIdentity as the container's hasher.
 */
struct KmerHashMix
{
    template <KmerType K>
    [[nodiscard]] std::size_t operator()(K kmer) const noexcept
    {
        return static_cast<std::size_t>(mix64_(kmer.value));
    }

    template <WideKmerType K>
    [[nodiscard]] std::size_t operator()(K const& kmer) const noexcept
    {
        std::uint64_t hash = 0;
        for (std::uint64_t const word : kmer.words) {
            hash = mix64_(hash ^ word);
        }
        return static_cast<std::size_t>(hash);
    }
};

} // namespace fisk

namespace std {

/**
 * @brief Make Kmer usable as a key in the standard unordered containers.
 *
 * Defaults to the mixing hash rather than the identity, so that the path a user gets without
 * thinking about it is the one that behaves. The cost is bounded and small; the cost of identity
 * hashing with clustered keys is neither.
 */
template <fisk::Encoding E, fisk::Layout L>
struct hash<fisk::Kmer<E, L>> : fisk::KmerHashMix
{};

/**
 * @brief Make SpacedKmer usable as a key in the standard unordered containers, as for Kmer.
 */
template <fisk::Encoding E, fisk::Layout L>
struct hash<fisk::SpacedKmer<E, L>> : fisk::KmerHashMix
{};

/**
 * @brief Make WideKmer usable as a key in the standard unordered containers, as for Kmer.
 */
template <fisk::Encoding E, fisk::Layout L, std::size_t Capacity>
struct hash<fisk::WideKmer<E, L, Capacity>> : fisk::KmerHashMix
{};

} // namespace std
