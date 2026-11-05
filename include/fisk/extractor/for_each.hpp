#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>

#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/types.hpp"
#include "fisk/extractor/internal.hpp"
#include "fisk/extractor/kmer_extractor.hpp"
#include "fisk/extractor/spaced_kmer_extractor.hpp"
#include "fisk/extractor/spaced_kmer_masks.hpp"

namespace fisk {

// =================================================================================================
//     For Each K-mer
// =================================================================================================

// The k-mer extraction with the estimate of KmerExtractor: the algorithm that is likely the
// fastest by what this build and CPU support, without tuning. Each overload takes one kind of
// input; the internal functions do the work for all of them.

template <InputFormat I, InputCheck C, Encoding E, Layout L, typename Seq, typename Callback>
FISK_ALWAYS_INLINE inline void for_each_kmer_estimate_(
    Seq const& seq, std::size_t k, Callback& callback
) {
    using Preference = typename KmerExtractorPreference<I, C, E, L>::type;
    auto const settings = make_kmer_extractor_settings_(k);
    visit_estimate<Preference, Callback>([&](auto const& candidate) FISK_ALWAYS_INLINE_LAMBDA {
        candidate.run(settings, seq, callback);
    });
}

template <InputFormat I, InputCheck C, Encoding E, Layout L, typename Callback>
inline std::string for_each_kmer_estimate_name_()
{
    using Preference = typename KmerExtractorPreference<I, C, E, L>::type;
    return visit_estimate<Preference, Callback>([](auto const& candidate) {
        return std::remove_cvref_t<decltype(candidate)>::name();
    });
}

// The checks of the input of the overloads for packed sequences. These overloads then run with the
// check that their input requires, so that a failed check is the only error.
template <InputCheck C>
inline constexpr void for_each_packed_unchecked_assert_()
{
    static_assert(
        C == InputCheck::kUnchecked,
        "A PackedSequence has no validity to check. Use InputCheck::kUnchecked, or a "
        "PackedSequenceWithValidity for checked input, see pack_sequence_checked()."
    );
}

template <InputCheck C>
inline constexpr void for_each_packed_checked_assert_()
{
    static_assert(
        C == InputCheck::kChecked,
        "A PackedSequenceWithValidity is for checked input. Use InputCheck::kChecked, or a "
        "PackedSequence for unchecked input, see pack_sequence_unchecked()."
    );
}

/**
 * @brief Call `callback` with each k-mer of `seq`, with the algorithm that KmerExtractor estimates
 * to be the fastest on this hardware, without tuning.
 *
 * This is the easy start: one call, with the choices that change the result as template
 * arguments, and the algorithm chosen by what this build and CPU support:
 *
 *     using namespace fisk;
 *     for_each_kmer<InputCheck::kChecked, Encoding::kACGT, Layout::kMSB>(
 *         seq, 31, [&](std::size_t pos, KmerAcgtMsb kmer) { ... }
 *     );
 *
 * The overloads for packed input take the Encoding, Layout, and InputCheck from the type of the
 * sequence. for_each_kmer_algorithm_name() names the algorithm. To measure the fastest algorithm
 * instead, see KmerExtractor; to call one of them directly, see the functions in kmer_extract/,
 * such as for_each_kmer_ascii_checked_chunked_scalar().
 *
 * The callback takes single k-mers, and can in addition take vectors of them, as for
 * KmerExtractor::for_each(). The k-mers are handed out in order of their positions.
 *
 * @tparam C InputCheck: kChecked skips the k-mers with invalid characters; kUnchecked trusts the
 *         input to be valid.
 * @tparam E Encoding of the k-mers.
 * @tparam L Layout of the k-mers.
 * @throws std::invalid_argument if `k` is not in [1, 32].
 */
template <InputCheck C, Encoding E, Layout L, typename Callback>
    requires KmerCallback<std::remove_reference_t<Callback>&, E, L>
inline void for_each_kmer(std::string_view seq, std::size_t k, Callback&& callback)
{
    for_each_kmer_estimate_<InputFormat::kAscii, C, E, L>(seq, k, callback);
}

/**
 * @brief Call `callback` with each k-mer of a PackedSequence.
 *
 * @see for_each_kmer(std::string_view, std::size_t, Callback&&) for ASCII input.
 */
template <InputCheck C = InputCheck::kUnchecked, Encoding E, Layout L, typename Callback>
    requires KmerCallback<std::remove_reference_t<Callback>&, E, L>
inline void for_each_kmer(PackedSequence<E, L> const& seq, std::size_t k, Callback&& callback)
{
    for_each_packed_unchecked_assert_<C>();
    for_each_kmer_estimate_<InputFormat::kPacked, InputCheck::kUnchecked, E, L>(seq, k, callback);
}

/**
 * @brief Call `callback` with each valid k-mer of a PackedSequenceWithValidity.
 *
 * @see for_each_kmer(std::string_view, std::size_t, Callback&&) for ASCII input.
 */
template <InputCheck C = InputCheck::kChecked, Encoding E, Layout L, typename Callback>
    requires KmerCallback<std::remove_reference_t<Callback>&, E, L>
inline void for_each_kmer(
    PackedSequenceWithValidity<E, L> const& seq, std::size_t k, Callback&& callback
) {
    for_each_packed_checked_assert_<C>();
    for_each_kmer_estimate_<InputFormat::kPacked, InputCheck::kChecked, E, L>(seq, k, callback);
}

/**
 * @brief The name of the algorithm that for_each_kmer() runs with these arguments, which are
 * not used otherwise: the name of the k-mer extraction function, followed by the SIMD arch, as in
 * KmerExtractor::algorithm_name().
 */
template <InputCheck C, Encoding E, Layout L, typename Callback>
    requires KmerCallback<std::remove_reference_t<Callback>&, E, L>
inline std::string for_each_kmer_algorithm_name(std::string_view, std::size_t, Callback&&)
{
    return for_each_kmer_estimate_name_<
        InputFormat::kAscii, C, E, L, std::remove_reference_t<Callback>
    >();
}

template <InputCheck C = InputCheck::kUnchecked, Encoding E, Layout L, typename Callback>
    requires KmerCallback<std::remove_reference_t<Callback>&, E, L>
inline std::string for_each_kmer_algorithm_name(
    PackedSequence<E, L> const&, std::size_t, Callback&&
) {
    for_each_packed_unchecked_assert_<C>();
    return for_each_kmer_estimate_name_<
        InputFormat::kPacked, InputCheck::kUnchecked, E, L, std::remove_reference_t<Callback>
    >();
}

template <InputCheck C = InputCheck::kChecked, Encoding E, Layout L, typename Callback>
    requires KmerCallback<std::remove_reference_t<Callback>&, E, L>
inline std::string for_each_kmer_algorithm_name(
    PackedSequenceWithValidity<E, L> const&, std::size_t, Callback&&
) {
    for_each_packed_checked_assert_<C>();
    return for_each_kmer_estimate_name_<
        InputFormat::kPacked, InputCheck::kChecked, E, L, std::remove_reference_t<Callback>
    >();
}

// =================================================================================================
//     For Each Spaced K-mer
// =================================================================================================

// The spaced k-mer extraction with the estimate of SpacedKmerExtractor, as for for_each_kmer().
// The overloads take masks of any Layout `ML`, so that masks of another Layout than the spaced
// k-mers fail here, with a message, instead of leaving no overload to match. The extraction then
// only runs with matching ones, so that a failed check is the only error.

template <Layout L, Layout ML>
inline constexpr void for_each_spaced_kmer_layout_assert_()
{
    static_assert(
        ML == L,
        "The masks are for another Layout than the spaced k-mers. Make them as SpacedKmerMasks<L> "
        "with the Layout of the extraction."
    );
}

template <
    InputFormat I, InputCheck C, Encoding E, Layout L, Layout ML, std::size_t MaxMasks,
    std::size_t MaxSpan, typename Seq, typename Callback
>
FISK_ALWAYS_INLINE inline void for_each_spaced_kmer_estimate_(
    Seq const& seq, SpacedKmerMasks<ML, MaxMasks, MaxSpan> const& masks, Callback& callback
) {
    for_each_spaced_kmer_layout_assert_<L, ML>();
    if constexpr (ML == L) {
        using Preference = typename SpacedKmerExtractorPreference<I, C, E, L, MaxMasks>::type;
        visit_estimate<Preference, Callback>([&](auto const& candidate) FISK_ALWAYS_INLINE_LAMBDA {
            candidate.run(masks, seq, callback);
        });
    }
}

template <
    InputFormat I, InputCheck C, Encoding E, Layout L, Layout ML, std::size_t MaxMasks,
    typename Callback
>
inline std::string for_each_spaced_kmer_estimate_name_()
{
    for_each_spaced_kmer_layout_assert_<L, ML>();
    if constexpr (ML == L) {
        using Preference = typename SpacedKmerExtractorPreference<I, C, E, L, MaxMasks>::type;
        return visit_estimate<Preference, Callback>([](auto const& candidate) {
            return std::remove_cvref_t<decltype(candidate)>::name();
        });
    } else {
        return {};
    }
}

/**
 * @brief Call `callback` with each spaced k-mer of `seq` for each of the `masks`, with the
 * algorithm that SpacedKmerExtractor estimates to be the fastest on this hardware, without tuning.
 *
 * As for_each_kmer(), for a prepared set of masks instead of a k-mer size. The set is prepared
 * once, and can then be used for as many sequences as needed:
 *
 *     using namespace fisk;
 *     SpacedKmerMasks<Layout::kMSB> const masks({
 *         "1101101101101101101101101101101",
 *         "1011011011011011011011011011011"
 *     });
 *     for_each_spaced_kmer<InputCheck::kChecked, Encoding::kACGT, Layout::kMSB>(
 *         seq, masks, [&](std::size_t pos, std::size_t mask_idx, SpacedKmerAcgtMsb kmer) { ... }
 *     );
 *
 * for_each_spaced_kmer_algorithm_name() names the algorithm. To measure the fastest algorithm
 * instead, see SpacedKmerExtractor; to call one of them directly, see the functions in
 * spaced_kmer_extract/.
 *
 * The callback takes single spaced k-mers with the index of their mask, and can in addition take
 * vectors of them, as for SpacedKmerExtractor::for_each(). The spaced k-mers of each mask are
 * handed out in order of their positions; the order between masks depends on the algorithm, see
 * SpacedKmerExtractor.
 *
 * @tparam C InputCheck: kChecked skips the spaced k-mers with invalid characters at the kept
 *         positions; kUnchecked trusts the input to be valid.
 * @tparam E Encoding of the spaced k-mers.
 * @tparam L Layout of the spaced k-mers, and of the masks.
 */
template <
    InputCheck C, Encoding E, Layout L, Layout ML, std::size_t MaxMasks, std::size_t MaxSpan,
    typename Callback
>
    requires SpacedKmerCallback<std::remove_reference_t<Callback>&, E, L, false>
inline void for_each_spaced_kmer(
    std::string_view seq, SpacedKmerMasks<ML, MaxMasks, MaxSpan> const& masks,
    Callback&& callback
) {
    for_each_spaced_kmer_estimate_<InputFormat::kAscii, C, E, L>(seq, masks, callback);
}

/**
 * @brief Call `callback` with each spaced k-mer of a PackedSequence.
 *
 * @see for_each_spaced_kmer(std::string_view, SpacedKmerMasks<ML, MaxMasks> const&, Callback&&)
 * for ASCII input.
 */
template <
    InputCheck C = InputCheck::kUnchecked, Encoding E, Layout L, Layout ML, std::size_t MaxMasks,
    std::size_t MaxSpan, typename Callback
>
    requires SpacedKmerCallback<std::remove_reference_t<Callback>&, E, L, false>
inline void for_each_spaced_kmer(
    PackedSequence<E, L> const& seq, SpacedKmerMasks<ML, MaxMasks, MaxSpan> const& masks,
    Callback&& callback
) {
    for_each_packed_unchecked_assert_<C>();
    for_each_spaced_kmer_estimate_<InputFormat::kPacked, InputCheck::kUnchecked, E, L>(
        seq, masks, callback
    );
}

/**
 * @brief Call `callback` with each valid spaced k-mer of a PackedSequenceWithValidity.
 *
 * @see for_each_spaced_kmer(std::string_view, SpacedKmerMasks<ML, MaxMasks> const&, Callback&&)
 * for ASCII input.
 */
template <
    InputCheck C = InputCheck::kChecked, Encoding E, Layout L, Layout ML, std::size_t MaxMasks,
    std::size_t MaxSpan, typename Callback
>
    requires SpacedKmerCallback<std::remove_reference_t<Callback>&, E, L, false>
inline void for_each_spaced_kmer(
    PackedSequenceWithValidity<E, L> const& seq,
    SpacedKmerMasks<ML, MaxMasks, MaxSpan> const& masks, Callback&& callback
) {
    for_each_packed_checked_assert_<C>();
    for_each_spaced_kmer_estimate_<InputFormat::kPacked, InputCheck::kChecked, E, L>(
        seq, masks, callback
    );
}

/**
 * @brief The name of the algorithm that for_each_spaced_kmer() runs with these arguments, which
 * are not used otherwise: the name of the spaced k-mer extraction function, followed by the bit
 * extraction, as in SpacedKmerExtractor::algorithm_name().
 */
template <
    InputCheck C, Encoding E, Layout L, Layout ML, std::size_t MaxMasks, std::size_t MaxSpan,
    typename Callback
>
    requires SpacedKmerCallback<std::remove_reference_t<Callback>&, E, L, false>
inline std::string for_each_spaced_kmer_algorithm_name(
    std::string_view, SpacedKmerMasks<ML, MaxMasks, MaxSpan> const&, Callback&&
) {
    return for_each_spaced_kmer_estimate_name_<
        InputFormat::kAscii, C, E, L, ML, MaxMasks, std::remove_reference_t<Callback>
    >();
}

template <
    InputCheck C = InputCheck::kUnchecked, Encoding E, Layout L, Layout ML, std::size_t MaxMasks,
    std::size_t MaxSpan, typename Callback
>
    requires SpacedKmerCallback<std::remove_reference_t<Callback>&, E, L, false>
inline std::string for_each_spaced_kmer_algorithm_name(
    PackedSequence<E, L> const&, SpacedKmerMasks<ML, MaxMasks, MaxSpan> const&, Callback&&
) {
    for_each_packed_unchecked_assert_<C>();
    return for_each_spaced_kmer_estimate_name_<
        InputFormat::kPacked, InputCheck::kUnchecked, E, L, ML, MaxMasks,
        std::remove_reference_t<Callback>
    >();
}

template <
    InputCheck C = InputCheck::kChecked, Encoding E, Layout L, Layout ML, std::size_t MaxMasks,
    std::size_t MaxSpan, typename Callback
>
    requires SpacedKmerCallback<std::remove_reference_t<Callback>&, E, L, false>
inline std::string for_each_spaced_kmer_algorithm_name(
    PackedSequenceWithValidity<E, L> const&, SpacedKmerMasks<ML, MaxMasks, MaxSpan> const&,
    Callback&&
) {
    for_each_packed_checked_assert_<C>();
    return for_each_spaced_kmer_estimate_name_<
        InputFormat::kPacked, InputCheck::kChecked, E, L, ML, MaxMasks,
        std::remove_reference_t<Callback>
    >();
}

} // namespace fisk
