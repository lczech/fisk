#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>

#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/types.hpp"
#include "fisk/extraction/estimate.hpp"
#include "fisk/extraction/kmer_estimate.hpp"
#include "fisk/extraction/spaced_kmer_estimate.hpp"
#include "fisk/extraction/spaced_kmer_mask_forms.hpp"

namespace fisk {

// =================================================================================================
//     For Each K-mer
// =================================================================================================

// The k-mer extraction with the estimate: the algorithm that is likely the fastest by what this
// build and CPU support, see KmerEstimate. Each overload takes one kind of input; the
// internal functions do the work for all of them.

template <InputFormat I, InputCheck C, Encoding E, Layout L, typename Seq, typename Callback>
FISK_ALWAYS_INLINE inline void for_each_kmer_estimate_(
    Seq const& seq, std::size_t k, Callback& callback
) {
    using Preference = typename KmerEstimate<I, C, E, L>::type;
    visit_estimate<Preference>([&](auto const& candidate) FISK_ALWAYS_INLINE_LAMBDA {
        candidate.run(k, seq, callback);
    });
}

template <InputFormat I, InputCheck C, Encoding E, Layout L>
inline std::string for_each_kmer_estimate_name_()
{
    using Preference = typename KmerEstimate<I, C, E, L>::type;
    return visit_estimate<Preference>([](auto const& candidate) {
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
 * @brief Call `callback` with each k-mer of `seq`, with the algorithm that is likely the fastest on
 * this hardware.
 *
 * One call, with the choices that change the result as template arguments, and the algorithm
 * chosen by what this build and CPU support, the estimate: the SIMD loop of the widest SIMD arch
 * of the build of AVX2, SSE2, and Neon, where the CPU has it, else the scalar loop, see
 * CandidateSimdArchs:
 *
 *     using namespace fisk;
 *     for_each_kmer<InputCheck::kChecked, Encoding::kACGT, Layout::kMSB>(
 *         seq, 31, [&](std::size_t pos, KmerAcgtMsb kmer) { ... }
 *     );
 *
 * The overloads for packed input take the Encoding, Layout, and InputCheck from the type of the
 * sequence. for_each_kmer_algorithm_name() names the algorithm. To measure the fastest algorithm
 * on your hardware, with your input and callback, see KmerExtractor. To call another algorithm,
 * see the functions in kmer_extract/, such as for_each_kmer_ascii_checked_chunked_scalar().
 *
 * The callback takes single k-mers, as `callback(pos, kmer)` or `callback(kmer)`, see
 * KmerCallback. It can in addition take the vectors of k-mers of the SIMD algorithms, see
 * KmerVectorCallback and KmerVectorValidLanesCallback, for instance as a callback_overload, or a
 * struct with several `operator()`. The SIMD algorithms then hand it their vectors, and else their
 * k-mers one by one; for the vectors of AVX512, call its functions directly, see
 * CandidateSimdArchs. Positions and counts are std::size_t, as sequences can exceed 2^32
 * nucleotides: a callback that takes them as a narrower integer, such as `int`, fails to compile,
 * where its type tells its parameter types, which a lambda with an `auto` parameter and a struct
 * with several `operator()` do not; take std::size_t (or `auto`) there as well. The callback is a
 * template parameter, so that it can be inlined into the loop; each callback type compiles all
 * algorithms that the estimate can choose.
 *
 * The k-mers are handed out in order of their positions. The function can run on several threads
 * at once; ASCII input is packed chunk by chunk into a buffer per thread.
 *
 * @tparam C InputCheck: kChecked skips the k-mers with invalid characters; kUnchecked trusts the
 *         input to be valid.
 * @tparam E Encoding of the k-mers.
 * @tparam L Layout of the k-mers.
 * @throws std::invalid_argument if `k` is not in [1, 32].
 */
template <InputCheck C, Encoding E, Layout L, typename Callback>
    requires KmerCallback<std::remove_reference_t<Callback>&, E, L>
FISK_ALWAYS_INLINE inline void for_each_kmer(
    std::string_view seq, std::size_t k, Callback&& callback
) {
    for_each_kmer_estimate_<InputFormat::kAscii, C, E, L>(seq, k, callback);
}

/**
 * @brief Call `callback` with each k-mer of a PackedSequence.
 *
 * @see for_each_kmer(std::string_view, std::size_t, Callback&&) for ASCII input.
 */
template <InputCheck C = InputCheck::kUnchecked, Encoding E, Layout L, typename Callback>
    requires KmerCallback<std::remove_reference_t<Callback>&, E, L>
FISK_ALWAYS_INLINE inline void for_each_kmer(
    PackedSequence<E, L> const& seq, std::size_t k, Callback&& callback
) {
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
FISK_ALWAYS_INLINE inline void for_each_kmer(
    PackedSequenceWithValidity<E, L> const& seq, std::size_t k, Callback&& callback
) {
    for_each_packed_checked_assert_<C>();
    for_each_kmer_estimate_<InputFormat::kPacked, InputCheck::kChecked, E, L>(seq, k, callback);
}

/**
 * @brief The name of the algorithm that for_each_kmer() runs with these arguments, which are
 * not used otherwise: the name of the k-mer extraction function, followed by the SIMD arch, such
 * as "for_each_kmer_ascii_checked_chunked_simd_avx2". The same for all callbacks.
 */
template <InputCheck C, Encoding E, Layout L>
inline std::string for_each_kmer_algorithm_name(std::string_view, std::size_t)
{
    return for_each_kmer_estimate_name_<InputFormat::kAscii, C, E, L>();
}

template <InputCheck C = InputCheck::kUnchecked, Encoding E, Layout L>
inline std::string for_each_kmer_algorithm_name(PackedSequence<E, L> const&, std::size_t)
{
    for_each_packed_unchecked_assert_<C>();
    return for_each_kmer_estimate_name_<InputFormat::kPacked, InputCheck::kUnchecked, E, L>();
}

template <InputCheck C = InputCheck::kChecked, Encoding E, Layout L>
inline std::string for_each_kmer_algorithm_name(
    PackedSequenceWithValidity<E, L> const&, std::size_t
) {
    for_each_packed_checked_assert_<C>();
    return for_each_kmer_estimate_name_<InputFormat::kPacked, InputCheck::kChecked, E, L>();
}

// =================================================================================================
//     For Each Spaced K-mer
// =================================================================================================

// The spaced k-mer extraction with the estimate, see SpacedKmerEstimate, as for
// for_each_kmer().
// The overloads take masks of any Layout `MasksLayout`, so that masks of another Layout than the
// spaced k-mers fail here, with a message, instead of leaving no overload to match. The extraction
// then only runs with matching ones, so that a failed check is the only error.

template <Layout L, Layout MasksLayout>
inline constexpr void for_each_spaced_kmer_layout_assert_()
{
    static_assert(
        MasksLayout == L,
        "The masks are for another Layout than the spaced k-mers. Make them as "
        "SpacedKmerMaskForms<L> with the Layout of the extraction."
    );
}

template <
    InputFormat I, InputCheck C, Encoding E, Layout L, Layout MasksLayout, std::size_t MaxMasks,
    std::size_t MaxSpan, typename Seq, typename Callback
>
FISK_ALWAYS_INLINE inline void for_each_spaced_kmer_estimate_(
    Seq const& seq, SpacedKmerMaskForms<MasksLayout, MaxMasks, MaxSpan> const& mask_forms,
    Callback& callback
) {
    for_each_spaced_kmer_layout_assert_<L, MasksLayout>();
    if constexpr (MasksLayout == L) {
        using Preference = typename SpacedKmerEstimate<I, C, E, L, MaxMasks>::type;
        visit_estimate<Preference>([&](auto const& candidate) FISK_ALWAYS_INLINE_LAMBDA {
            candidate.run(mask_forms, seq, callback);
        });
    }
}

template <
    InputFormat I, InputCheck C, Encoding E, Layout L, Layout MasksLayout, std::size_t MaxMasks
>
inline std::string for_each_spaced_kmer_estimate_name_()
{
    for_each_spaced_kmer_layout_assert_<L, MasksLayout>();
    if constexpr (MasksLayout == L) {
        using Preference = typename SpacedKmerEstimate<I, C, E, L, MaxMasks>::type;
        return visit_estimate<Preference>([](auto const& candidate) {
            return std::remove_cvref_t<decltype(candidate)>::name();
        });
    } else {
        return {};
    }
}

/**
 * @brief Call `callback` with each spaced k-mer of `seq` for each of the masks of `mask_forms`,
 * with the algorithm that is likely the fastest on this hardware.
 *
 * As for_each_kmer(), for a set of spaced k-mer masks instead of a k-mer size, prepared in all
 * forms of the algorithms, see SpacedKmerMaskForms. The set is prepared once, and can then be used
 * for as many sequences as needed:
 *
 *     using namespace fisk;
 *     SpacedKmerMaskForms<Layout::kMSB> const mask_forms({
 *         "1101101101101101101101101101101",
 *         "1011011011011011011011011011011"
 *     });
 *     for_each_spaced_kmer<InputCheck::kChecked, Encoding::kACGT, Layout::kMSB>(
 *         seq, mask_forms,
 *         [&](std::size_t pos, std::size_t mask_idx, SpacedKmerAcgtMsb kmer) { ... }
 *     );
 *
 * The estimate is the scalar loop with PEXT, unless the CPU is known for a slow PEXT, see
 * pext_likely_fast(), else the SIMD loop with the butterfly bit extraction of the SIMD arch of
 * for_each_kmer(), where the CPU has it, else the scalar loop with the butterfly.
 * for_each_spaced_kmer_algorithm_name() names it. To measure the fastest algorithm, see
 * SpacedKmerExtractor. To call another algorithm, see the functions in spaced_kmer_extract/.
 *
 * The callback takes single spaced k-mers, as `callback(pos, mask_idx, spaced_kmer)`. It can in
 * addition take the vectors of the SIMD algorithms, as
 * `callback(pos, mask_idx, spaced_kmers, valid_count)` when unchecked, or
 * `callback(pos, mask_idx, spaced_kmers, valid_lanes)` when checked, see SpacedKmerVectorCallback
 * and SpacedKmerVectorValidLanesCallback. The SIMD algorithms then hand it their vectors, and else
 * their spaced k-mers one by one. As the estimate prefers PEXT, which hands out single spaced
 * k-mers, a callback that takes vectors can be faster with a SIMD algorithm called directly on some
 * CPUs. Positions, mask indices, and counts are std::size_t, see for_each_kmer().
 *
 * The spaced k-mers of each mask are handed out in order of their positions; the order between
 * masks differs between the algorithms: the scalar loops hand out the spaced k-mers of all masks
 * at one position before going on to the next position, while the SIMD loops hand out those of
 * one mask for a whole vector of positions, then those of the next mask for the same positions.
 * The callback is compiled for each algorithm that the estimate can choose, and each number of
 * masks up to the `MaxMasks` of the SpacedKmerMaskForms; a smaller `MaxMasks`, such as the number
 * of masks that are used, hence shortens the compile time.
 *
 * @tparam C InputCheck: kChecked skips the spaced k-mers with invalid characters at the kept
 *         positions; kUnchecked trusts the input to be valid.
 * @tparam E Encoding of the spaced k-mers.
 * @tparam L Layout of the spaced k-mers, and of the masks.
 */
template <
    InputCheck C, Encoding E, Layout L, Layout MasksLayout, std::size_t MaxMasks,
    std::size_t MaxSpan, typename Callback
>
    requires SpacedKmerCallback<std::remove_reference_t<Callback>&, E, L, false>
FISK_ALWAYS_INLINE inline void for_each_spaced_kmer(
    std::string_view seq, SpacedKmerMaskForms<MasksLayout, MaxMasks, MaxSpan> const& mask_forms,
    Callback&& callback
) {
    for_each_spaced_kmer_estimate_<InputFormat::kAscii, C, E, L>(seq, mask_forms, callback);
}

/**
 * @brief Call `callback` with each spaced k-mer of a PackedSequence.
 *
 * @see for_each_spaced_kmer(std::string_view, SpacedKmerMaskForms const&, Callback&&) for ASCII
 * input.
 */
template <
    InputCheck C = InputCheck::kUnchecked, Encoding E, Layout L, Layout MasksLayout,
    std::size_t MaxMasks, std::size_t MaxSpan, typename Callback
>
    requires SpacedKmerCallback<std::remove_reference_t<Callback>&, E, L, false>
FISK_ALWAYS_INLINE inline void for_each_spaced_kmer(
    PackedSequence<E, L> const& seq,
    SpacedKmerMaskForms<MasksLayout, MaxMasks, MaxSpan> const& mask_forms, Callback&& callback
) {
    for_each_packed_unchecked_assert_<C>();
    for_each_spaced_kmer_estimate_<InputFormat::kPacked, InputCheck::kUnchecked, E, L>(
        seq, mask_forms, callback
    );
}

/**
 * @brief Call `callback` with each valid spaced k-mer of a PackedSequenceWithValidity.
 *
 * @see for_each_spaced_kmer(std::string_view, SpacedKmerMaskForms const&, Callback&&) for ASCII
 * input.
 */
template <
    InputCheck C = InputCheck::kChecked, Encoding E, Layout L, Layout MasksLayout,
    std::size_t MaxMasks, std::size_t MaxSpan, typename Callback
>
    requires SpacedKmerCallback<std::remove_reference_t<Callback>&, E, L, false>
FISK_ALWAYS_INLINE inline void for_each_spaced_kmer(
    PackedSequenceWithValidity<E, L> const& seq,
    SpacedKmerMaskForms<MasksLayout, MaxMasks, MaxSpan> const& mask_forms, Callback&& callback
) {
    for_each_packed_checked_assert_<C>();
    for_each_spaced_kmer_estimate_<InputFormat::kPacked, InputCheck::kChecked, E, L>(
        seq, mask_forms, callback
    );
}

/**
 * @brief The name of the algorithm that for_each_spaced_kmer() runs with these arguments, which
 * are not used otherwise: the name of the spaced k-mer extraction function, followed by the bit
 * extraction, such as "for_each_spaced_kmer_ascii_checked_chunked_scalar_pext". The same for all
 * callbacks.
 */
template <
    InputCheck C, Encoding E, Layout L, Layout MasksLayout, std::size_t MaxMasks,
    std::size_t MaxSpan
>
inline std::string for_each_spaced_kmer_algorithm_name(
    std::string_view, SpacedKmerMaskForms<MasksLayout, MaxMasks, MaxSpan> const&
) {
    return for_each_spaced_kmer_estimate_name_<
        InputFormat::kAscii, C, E, L, MasksLayout, MaxMasks
    >();
}

template <
    InputCheck C = InputCheck::kUnchecked, Encoding E, Layout L, Layout MasksLayout,
    std::size_t MaxMasks, std::size_t MaxSpan
>
inline std::string for_each_spaced_kmer_algorithm_name(
    PackedSequence<E, L> const&, SpacedKmerMaskForms<MasksLayout, MaxMasks, MaxSpan> const&
) {
    for_each_packed_unchecked_assert_<C>();
    return for_each_spaced_kmer_estimate_name_<
        InputFormat::kPacked, InputCheck::kUnchecked, E, L, MasksLayout, MaxMasks
    >();
}

template <
    InputCheck C = InputCheck::kChecked, Encoding E, Layout L, Layout MasksLayout,
    std::size_t MaxMasks, std::size_t MaxSpan
>
inline std::string for_each_spaced_kmer_algorithm_name(
    PackedSequenceWithValidity<E, L> const&,
    SpacedKmerMaskForms<MasksLayout, MaxMasks, MaxSpan> const&
) {
    for_each_packed_checked_assert_<C>();
    return for_each_spaced_kmer_estimate_name_<
        InputFormat::kPacked, InputCheck::kChecked, E, L, MasksLayout, MaxMasks
    >();
}

} // namespace fisk
