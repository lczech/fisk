#include <cstddef>
#include <cstdint>
#include <string>

#include "fisk/core/kmer_callback.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers
// =================================================================================================

// The vector forms are checked with the words of SimdScalar: spelling out a vector type such as
// __m256i as a template argument makes GCC warn about its ignored attributes.
using Vec = std::uint64_t;
using K = Kmer<Encoding::kACGT, Layout::kMSB>;
using S = SpacedKmer<Encoding::kACGT, Layout::kMSB>;

template <typename T>
concept KmerCb = KmerCallback<T, Encoding::kACGT, Layout::kMSB>;
template <typename T>
concept SpacedCb = SpacedKmerCallback<T, Encoding::kACGT, Layout::kMSB, false>;
template <typename T>
concept SpacedSingleCb = SpacedKmerCallback<T, Encoding::kACGT, Layout::kMSB, true>;

// =================================================================================================
//     Contiguous K-mers
// =================================================================================================

static_assert(KmerCb<decltype([](std::size_t, K const&) {})>);
static_assert(KmerCb<decltype([](K const&) {})>);
static_assert(KmerCb<decltype([](auto const&) {})>);
static_assert(!KmerCb<decltype([](std::size_t, std::size_t, K const&) {})>);
static_assert(!KmerCb<decltype([](std::string const&) {})>);
static_assert(!KmerCb<decltype([](std::size_t, S const&) {})>);

static_assert(KmerVectorCallback<decltype([](std::size_t, Vec, std::size_t) {}), Vec>);
static_assert(KmerVectorCallback<decltype([](Vec, std::size_t) {}), Vec>);
static_assert(!KmerVectorCallback<decltype([](Vec) {}), Vec>);
static_assert(!KmerVectorCallback<
    decltype([](std::size_t, std::size_t, Vec, std::size_t) {}), Vec
>);

// =================================================================================================
//     Spaced K-mers
// =================================================================================================

// The shorthand without mask_idx is only accepted for a single mask.
static_assert(SpacedCb<decltype([](std::size_t, std::size_t, S const&) {})>);
static_assert(SpacedSingleCb<decltype([](std::size_t, std::size_t, S const&) {})>);
static_assert(!SpacedCb<decltype([](std::size_t, S const&) {})>);
static_assert(SpacedSingleCb<decltype([](std::size_t, S const&) {})>);
static_assert(!SpacedSingleCb<decltype([](S const&) {})>);
static_assert(!SpacedCb<decltype([](std::size_t, std::size_t, K const&) {})>);

// Only the form with mask_idx, and the scalar forms do not pass as vector forms, nor vice versa.
static_assert(SpacedKmerVectorCallback<
    decltype([](std::size_t, std::size_t, Vec, std::size_t) {}), Vec
>);
static_assert(!SpacedKmerVectorCallback<decltype([](std::size_t, Vec, std::size_t) {}), Vec>);
static_assert(!SpacedKmerVectorCallback<
    decltype([](std::size_t, std::size_t, S const&) {}), Vec
>);
static_assert(!SpacedSingleCb<decltype([](std::size_t, std::size_t, Vec, std::size_t) {})>);
