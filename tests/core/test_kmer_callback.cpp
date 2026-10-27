#include <cstddef>
#include <cstdint>
#include <string>

#include "fisk/core/kmer_callback.hpp"
#include "fisk/simd/scalar.hpp"
#include "fisk/core/types.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers
// =================================================================================================

// The vector forms are checked with the words of SimdScalar, whose raw vector `Vec` has the type of
// a raw k-mer: spelling out a vector type such as __m256i as a template argument makes GCC warn
// about its ignored attributes.
using Vec = std::uint64_t;
using K = Kmer<Encoding::kACGT, Layout::kMSB>;
using S = SpacedKmer<Encoding::kACGT, Layout::kMSB>;
using KV = KmerVector<Encoding::kACGT, Layout::kMSB, SimdScalar>;
using SV = SpacedKmerVector<Encoding::kACGT, Layout::kMSB, SimdScalar>;
using Lanes = SimdVectorValidLanes<SimdScalar>;

template <typename T>
concept KmerCb = KmerCallback<T, Encoding::kACGT, Layout::kMSB>;
template <typename T>
concept VectorCb = KmerVectorCallback<T, Encoding::kACGT, Layout::kMSB, SimdScalar>;
template <typename T>
concept LanesCb = KmerVectorValidLanesCallback<T, Encoding::kACGT, Layout::kMSB, SimdScalar>;
template <typename T>
concept SpacedCb = SpacedKmerCallback<T, Encoding::kACGT, Layout::kMSB, false>;
template <typename T>
concept SpacedSingleCb = SpacedKmerCallback<T, Encoding::kACGT, Layout::kMSB, true>;
template <typename T>
concept SpacedVectorCb =
    SpacedKmerVectorCallback<T, Encoding::kACGT, Layout::kMSB, SimdScalar, false>;
template <typename T>
concept SpacedVectorSingleCb =
    SpacedKmerVectorCallback<T, Encoding::kACGT, Layout::kMSB, SimdScalar, true>;
template <typename T>
concept SpacedLanesCb =
    SpacedKmerVectorValidLanesCallback<T, Encoding::kACGT, Layout::kMSB, SimdScalar, false>;
template <typename T>
concept SpacedLanesSingleCb =
    SpacedKmerVectorValidLanesCallback<T, Encoding::kACGT, Layout::kMSB, SimdScalar, true>;

// =================================================================================================
//     Contiguous K-mers
// =================================================================================================

static_assert(KmerCb<decltype([](std::size_t, K const&) {})>);
static_assert(KmerCb<decltype([](K const&) {})>);
static_assert(KmerCb<decltype([](auto const&) {})>);
static_assert(!KmerCb<decltype([](std::size_t, std::size_t, K const&) {})>);
static_assert(!KmerCb<decltype([](std::string const&) {})>);
static_assert(!KmerCb<decltype([](std::size_t, S const&) {})>);

static_assert(VectorCb<decltype([](std::size_t, KV, std::size_t) {})>);
static_assert(!VectorCb<decltype([](KV, std::size_t) {})>);
static_assert(!VectorCb<decltype([](KV) {})>);
static_assert(!VectorCb<decltype([](std::size_t, std::size_t, KV, std::size_t) {})>);

// The vectors are tagged as the k-mers are: a callback for raw vectors, or for vectors of other
// conventions, does not take them.
static_assert(!VectorCb<decltype([](std::size_t, Vec, std::size_t) {})>);
static_assert(!VectorCb<decltype(
    [](std::size_t, KmerVector<Encoding::kACTG, Layout::kMSB, SimdScalar>, std::size_t) {}
)>);
static_assert(!VectorCb<decltype(
    [](std::size_t, KmerVector<Encoding::kACGT, Layout::kLSB, SimdScalar>, std::size_t) {}
)>);

// The valid lanes are a type of their own, so that a callback for a count of valid lanes does not
// take them, which the mask register of AVX512 would convert to.
static_assert(LanesCb<decltype([](std::size_t, KV, Lanes) {})>);
static_assert(!LanesCb<decltype([](KV, Lanes) {})>);
static_assert(!LanesCb<decltype([](std::size_t, KV, std::size_t) {})>);
static_assert(!LanesCb<decltype([](std::size_t, Vec, Lanes) {})>);

// The forms of single k-mers and of vectors differ in their number of arguments, so that generic
// lambdas are only one of them.
static_assert(KmerCb<decltype([](auto, auto) {})>);
static_assert(!LanesCb<decltype([](auto, auto) {})>);
static_assert(!KmerCb<decltype([](auto, auto, auto) {})>);
static_assert(LanesCb<decltype([](auto, auto, auto) {})>);
static_assert(VectorCb<decltype([](auto, auto, auto) {})>);

// =================================================================================================
//     Spaced K-mers
// =================================================================================================

// The shorthands without mask_idx, with or without pos, are only accepted for a single mask.
static_assert(SpacedCb<decltype([](std::size_t, std::size_t, S const&) {})>);
static_assert(SpacedSingleCb<decltype([](std::size_t, std::size_t, S const&) {})>);
static_assert(!SpacedCb<decltype([](std::size_t, S const&) {})>);
static_assert(SpacedSingleCb<decltype([](std::size_t, S const&) {})>);
static_assert(!SpacedCb<decltype([](S const&) {})>);
static_assert(SpacedSingleCb<decltype([](S const&) {})>);
static_assert(!SpacedSingleCb<decltype([](K const&) {})>);
static_assert(!SpacedCb<decltype([](std::size_t, std::size_t, K const&) {})>);

// The same holds for the vector forms.
static_assert(SpacedVectorCb<decltype([](std::size_t, std::size_t, SV, std::size_t) {})>);
static_assert(SpacedVectorSingleCb<decltype([](std::size_t, std::size_t, SV, std::size_t) {})>);
static_assert(!SpacedVectorCb<decltype([](std::size_t, SV, std::size_t) {})>);
static_assert(SpacedVectorSingleCb<decltype([](std::size_t, SV, std::size_t) {})>);

// The vectors are tagged: a callback for raw vectors, or for vectors of k-mers, does not take them.
// That keeps the vector shorthand apart from the scalar form with mask_idx, even for SimdScalar,
// whose raw vectors are words, and the scalar and vector forms do not pass as each other.
static_assert(!SpacedVectorCb<decltype([](std::size_t, std::size_t, Vec, std::size_t) {})>);
static_assert(!SpacedVectorSingleCb<decltype([](std::size_t, Vec, std::size_t) {})>);
static_assert(!SpacedVectorCb<decltype([](std::size_t, std::size_t, KV, std::size_t) {})>);
static_assert(!SpacedVectorSingleCb<decltype([](std::size_t, std::size_t, S const&) {})>);
static_assert(!SpacedSingleCb<decltype([](std::size_t, SV, std::size_t) {})>);
static_assert(!SpacedSingleCb<decltype([](std::size_t, std::size_t, SV, std::size_t) {})>);

// The same for the vector forms with valid lanes, which take neither a count of valid lanes nor
// vectors of k-mers.
static_assert(SpacedLanesCb<decltype([](std::size_t, std::size_t, SV, Lanes) {})>);
static_assert(SpacedLanesSingleCb<decltype([](std::size_t, std::size_t, SV, Lanes) {})>);
static_assert(!SpacedLanesCb<decltype([](std::size_t, SV, Lanes) {})>);
static_assert(SpacedLanesSingleCb<decltype([](std::size_t, SV, Lanes) {})>);
static_assert(!SpacedLanesCb<decltype([](std::size_t, std::size_t, SV, std::size_t) {})>);
static_assert(!SpacedLanesSingleCb<decltype([](std::size_t, SV, std::size_t) {})>);
static_assert(!SpacedLanesCb<decltype([](std::size_t, std::size_t, KV, Lanes) {})>);
static_assert(!SpacedVectorCb<decltype([](std::size_t, std::size_t, SV, Lanes) {})>);
