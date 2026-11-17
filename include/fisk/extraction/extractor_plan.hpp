#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "fisk/util/intrinsics.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/types.hpp"
#include "fisk/extraction/estimate.hpp"
#include "fisk/extraction/tune_options.hpp"

namespace fisk {

// =================================================================================================
//     Extractor Types
// =================================================================================================

/**
 * @brief The input sequence of an extractor: a `std::string_view` for ASCII input, else a
 * PackedSequence, or, to be checked, a PackedSequenceWithValidity.
 */
template <InputFormat I, InputCheck C, Encoding E, Layout L>
using ExtractorSequence = std::conditional_t<
    I == InputFormat::kAscii,
    std::string_view,
    std::conditional_t<
        C == InputCheck::kChecked, PackedSequenceWithValidity<E, L>, PackedSequence<E, L>
    >
>;

// =================================================================================================
//     Packed Pieces
// =================================================================================================

/**
 * @brief The unit of the pieces of packed sequences, see ExtractorPackedPiece, in nucleotides.
 */
inline constexpr std::size_t kExtractorPieceUnit = 256;

/**
 * @brief A piece of a PackedSequence, without a copy: a PackedSequenceType, which the packed
 * algorithms take as they take the sequence. Made by make_extractor_packed_piece().
 */
template <Encoding E, Layout L>
class ExtractorPackedPiece
{
public:

    static constexpr Encoding encoding = E;
    static constexpr Layout layout = L;

    ExtractorPackedPiece(std::uint8_t const* codes, std::size_t length) noexcept
        : codes_(codes)
        , length_(length)
    {}

    std::size_t length() const noexcept
    {
        return length_;
    }

    std::span<std::uint8_t const> codes() const noexcept
    {
        return {codes_, (length_ + 3) / 4};
    }

private:

    std::uint8_t const* codes_;
    std::size_t length_;
};

/**
 * @brief A piece of a PackedSequenceWithValidity, without a copy: a
 * PackedSequenceWithValidityType. Made by make_extractor_packed_piece().
 */
template <Encoding E, Layout L>
class ExtractorPackedPieceWithValidity
{
public:

    static constexpr Encoding encoding = E;
    static constexpr Layout layout = L;

    ExtractorPackedPieceWithValidity(
        ExtractorPackedPiece<E, L> sequence, std::uint8_t const* validity
    ) noexcept
        : sequence_(sequence)
        , validity_(validity)
    {}

    std::size_t length() const noexcept
    {
        return sequence_.length();
    }

    std::span<std::uint8_t const> codes() const noexcept
    {
        return sequence_.codes();
    }

    std::span<std::uint8_t const> validity() const noexcept
    {
        return {validity_, (length() + 7) / 8};
    }

    // A reference to a member, as the algorithms keep the reference, see without_validity_().
    ExtractorPackedPiece<E, L> const& sequence() const noexcept
    {
        return sequence_;
    }

private:

    ExtractorPackedPiece<E, L> sequence_;
    std::uint8_t const* validity_;
};

/**
 * @brief The piece of the packed sequence `seq` with the k-mers that start at the `starts`
 * positions from `start`, and the `overlap` nucleotides after them that the last of those k-mers
 * reach (k - 1, or the span - 1 of spaced k-mers), or up to the end of the sequence.
 *
 * `start` is a multiple of kExtractorPieceUnit, and so is `starts`, unless the piece reaches the
 * end of the sequence. That keeps the algorithms within the k-mers of the piece:
 *
 *  - Their loops leave out the k-mers after the last one of a sequence only by its validity bits
 *    after the end, which are zero there, but are those of the following nucleotides in a piece.
 *    A number of starts that is a multiple of kExtractorPieceUnit ends the piece at a whole block
 *    of their loops (8 bytes of codes, see run_packed_bytewise_blocks_()), so that their last
 *    steps hold no position after its last k-mer.
 *  - They read whole words up into the padding after the codes and the validity bits, which in a
 *    piece are the following ones of the sequence, or its padding.
 *  - The codes of a piece start at a multiple of 64 bytes from those of the sequence, and its
 *    validity bits at a multiple of 32 bytes; the loops load both unaligned.
 */
template <PackedSequenceType Seq>
auto make_extractor_packed_piece(
    Seq const& seq, std::size_t start, std::size_t starts, std::size_t overlap
) noexcept
{
    assert(start % kExtractorPieceUnit == 0 && start <= seq.length());
    assert(starts % kExtractorPieceUnit == 0 || start + starts + overlap >= seq.length());
    std::size_t const length = std::min(starts + overlap, seq.length() - start);
    ExtractorPackedPiece<Seq::encoding, Seq::layout> const codes(
        seq.codes().data() + start / 4, length
    );
    if constexpr (PackedSequenceWithValidityType<Seq>) {
        return ExtractorPackedPieceWithValidity<Seq::encoding, Seq::layout>(
            codes, seq.validity().data() + start / 8
        );
    } else {
        return codes;
    }
}

// =================================================================================================
//     Extractor Plan
// =================================================================================================

/**
 * @brief An atomic that copies by value, so that a class with it stays copyable. Loads are
 * relaxed, and stores release, unless given otherwise.
 */
template <typename T>
class CopyableAtomic
{
public:

    CopyableAtomic(T value) noexcept
        : value_(value)
    {}

    CopyableAtomic(CopyableAtomic const& other) noexcept
        : value_(other.load())
    {}

    CopyableAtomic& operator=(CopyableAtomic const& other) noexcept
    {
        store(other.load());
        return *this;
    }

    T load(std::memory_order order = std::memory_order_relaxed) const noexcept
    {
        return value_.load(order);
    }

    void store(T value, std::memory_order order = std::memory_order_release) noexcept
    {
        value_.store(value, order);
    }

    bool compare_exchange_strong(T& expected, T desired) noexcept
    {
        return value_.compare_exchange_strong(expected, desired, std::memory_order_acq_rel);
    }

private:

    std::atomic<T> value_;
};

/**
 * @brief The callback of ExtractorPlan::for_each() with the positions shifted by `offset`, for the
 * pieces of a sequence. A call form is offered only where the callback has it, so that the
 * algorithms see the same forms, such as those for vectors, as with the callback itself; the forms
 * without a position, such as `callback(kmer)`, are passed on as they are.
 */
template <typename Callback>
struct ExtractorOffsetCallback
{
    Callback& callback;
    std::size_t offset;

    template <typename... Args>
        requires std::invocable<Callback&, std::size_t, Args...>
    FISK_ALWAYS_INLINE decltype(auto) operator()(std::size_t pos, Args&&... args)
    {
        // `callback` is checked here, as the algorithms see only this callback, see
        // check_callback_parameters_() (core/kmer_callback.hpp).
        check_callback_parameters_<Callback>();
        return callback(pos + offset, std::forward<Args>(args)...);
    }

    // The form without a position, `callback(kmer)`, the only one: of one argument, so that the
    // calls with a position never test the callback with a position too many, which would
    // instantiate generic lambdas of the callback, such as those for vectors, with wrong types.
    template <typename Arg>
        requires (!std::invocable<Callback&, std::size_t, Arg> && std::invocable<Callback&, Arg>)
    FISK_ALWAYS_INLINE decltype(auto) operator()(Arg&& arg)
    {
        return callback(std::forward<Arg>(arg));
    }
};

// The number of nucleotides after the start of the last k-mer of a piece of a sequence that the
// k-mer of size `k` reaches, see ExtractorPlan; for spaced k-mers, see SpacedKmerExtractor.
inline std::size_t extractor_overlap_(std::size_t k) noexcept
{
    return k - 1;
}

/**
 * @brief The phases of the in-situ tuning of an ExtractorPlan: off; armed by tune(), until
 * a call of for_each() claims it; claimed by a thread, which sets itself as its owner; and running,
 * in the calls of for_each() of its owner.
 */
enum class ExtractorTuningPhase
{
    kOff,
    kArmed,
    kClaiming,
    kRunning
};

/**
 * @brief The shared logic of KmerExtractor and SpacedKmerExtractor: the candidate algorithms, the
 * choice between them, and running the chosen one.
 *
 * The candidates are those of TypeList (extraction/estimate.hpp), run with the `Settings` as their
 * parameter, the k-mer size or the masks, on a sequence `S`: `Seq` for ASCII input, and a piece of
 * it for packed input, see make_extractor_packed_piece().
 *
 * The plan holds the index of its candidate, 1 + its position among the candidates, or 0 for the
 * estimate, the first candidate of the `Preference` that is preferred on this CPU, see
 * visit_estimate().
 *
 * The in-situ tuning, see tune(), times the candidates in the calls of for_each() itself,
 * in turns on pieces of their sequences, with their callbacks, and keeps the fastest. The index is
 * an atomic, so that the tuning can set it while other threads run for_each().
 */
template <typename Settings, typename Seq, typename Candidates, typename Preference>
class ExtractorPlan;

template <typename Settings, typename Seq, typename... Candidates, typename Preference>
class ExtractorPlan<Settings, Seq, TypeList<Candidates...>, Preference>
{
public:

    /**
     * @brief Prepare the plan for the `settings`, with the estimate, and armed for the in-situ
     * tuning with the default options, see tune().
     */
    explicit ExtractorPlan(Settings settings)
        : settings_(std::move(settings))
    {
        tune(TuneOptions{});
    }

    Settings const& settings() const noexcept
    {
        return settings_;
    }

    // -------------------------------------------------------------------------
    //     Run
    // -------------------------------------------------------------------------

    /**
     * @brief Run the candidate of the plan on `seq`; while the in-situ tuning runs in the calls of
     * this thread, the candidates in turn on its pieces, see tune().
     */
    template <typename Callback>
    FISK_ALWAYS_INLINE void for_each(Seq const& seq, Callback& callback) const
    {
        // The candidate to run: the one that the plan holds, or for the estimate, the first one of
        // the preference list that is preferred on this CPU, see visit_estimate(). So the estimate
        // runs the code of that candidate, instead of a copy of its own.
        std::size_t index = index_.load();
        if (index == 0) {
            index = estimate_index_();
        }

        // One loop with one jump to the candidates for the tuning and for the extraction, so that
        // both run the same machine code. Without tuning, the loop runs once, on all of the
        // sequence, or on the rest of it after the piece that ended the tuning. While tuning, each
        // piece has up to `piece_starts` k-mers, and runs the candidate whose turn it is, see
        // record_() for its timing. The positions of a piece are shifted by its offset in the
        // sequence, which is 0 without tuning.
        bool tuning = tuning_.phase.load() != ExtractorTuningPhase::kOff && claim_();
        std::size_t const total = kmer_starts_(seq);
        std::size_t at = 0;
        while (true) {
            std::size_t starts = total - at;
            if (tuning) [[unlikely]] {
                index = tuning_.order[tuning_.turn % tuning_.count];
                starts = std::min(starts, tuning_.piece_starts);
                if (tuning_.turn_starts == 0) {
                    tuning_.turn_start = std::chrono::steady_clock::now();
                }
            }
            auto const part = piece_(seq, at, starts);
            ExtractorOffsetCallback<Callback> shifted{callback, at};

            // At runtime: the jump to that candidate, with one branch per candidate. Not
            // std::visit, which calls the function for an alternative out of line, so that the
            // callback would escape into that call, and its state could not stay in registers in
            // the loops of the candidate.
            // The candidates are empty structs, see above, so each branch runs one of its own.
            [&]<std::size_t... I>(std::index_sequence<I...>) FISK_ALWAYS_INLINE_LAMBDA {
                (void) ((
                    index == I + 1 && (Candidates{}.run(settings_, part, shifted), true)
                ) || ...);
            }(std::index_sequence_for<Candidates...>{});

            if (!tuning) [[likely]] {
                break;
            }
            tuning = record_(starts);
            at += starts;
            if (at >= total) {
                break;
            }
            index = index_.load();
        }
    }

    // -------------------------------------------------------------------------
    //     Algorithm Names
    // -------------------------------------------------------------------------

    std::string algorithm_name() const
    {
        std::size_t const index = index_.load();
        if (index == 0) {
            return visit_estimate<Preference>([](auto const& preferred) {
                return std::remove_cvref_t<decltype(preferred)>::name();
            });
        }
        std::string name;
        std::size_t i = 0;
        ((++i == index ? (void) (name = Candidates::name()) : (void) 0), ...);
        return name;
    }

    std::vector<std::string> algorithm_names() const
    {
        std::vector<std::string> names;
        (
            [&] {
                if (Candidates::available()) {
                    names.push_back(Candidates::name());
                }
            }(),
            ...
        );
        return names;
    }

    void set_algorithm(std::string_view name)
    {
        std::size_t found = 0;
        std::size_t i = 0;
        (
            [&] {
                ++i;
                if (found == 0 && Candidates::name() == name && Candidates::available()) {
                    found = i;
                }
            }(),
            ...
        );
        if (found == 0) {
            throw std::invalid_argument(
                "set_algorithm(): unknown or unavailable algorithm \"" + std::string(name) + "\""
            );
        }
        tuning_.phase.store(ExtractorTuningPhase::kOff);
        index_.store(found);
    }

    // -------------------------------------------------------------------------
    //     Tune
    // -------------------------------------------------------------------------

    void estimate()
    {
        tuning_.phase.store(ExtractorTuningPhase::kOff);
        index_.store(0);
    }

    /**
     * @brief Tune in situ: the next calls of for_each() time the available candidates, see
     * TuneOptions, and then keep the fastest.
     *
     * Each candidate runs in turns of `time_per_algorithm / rounds`, on pieces of up to
     * `piece_length` k-mers of the sequences of those calls, or on whole sequences when they are
     * shorter; a turn ends after its time ran out, see record_(), and with a time of 0 or less,
     * after each piece. The turns go round the candidates `rounds` times, starting with the
     * estimate's candidate; the fastest time per k-mer of each candidate in a turn counts.
     * The first thread that calls for_each() afterwards tunes, in its calls; until it is done, the
     * calls of other threads run the plan as it was. Like the other choices, this call itself
     * must not run at the same time as for_each().
     */
    void tune(TuneOptions const& options)
    {
        tuning_.phase.store(ExtractorTuningPhase::kOff);

        // The options as TuneOptions documents them: `rounds` in [1, kTuneMaxRounds], so that the
        // number of turns and the time per turn cannot overflow, and `piece_length` in
        // [1, kTuneMaxPieceLength], rounded up to whole units, so that a turn reads the clock at
        // least every kTuneMaxPieceLength k-mers. A time of 0 or less ends each turn after a
        // piece, see record_().
        auto const rounds = std::clamp<std::size_t>(options.rounds, 1, kTuneMaxRounds);
        auto const length = std::clamp<std::size_t>(options.piece_length, 1, kTuneMaxPieceLength);
        tuning_.turn_time = options.time_per_algorithm / static_cast<std::int64_t>(rounds);
        tuning_.rounds = rounds;
        tuning_.piece_starts =
            (length + kExtractorPieceUnit - 1) / kExtractorPieceUnit * kExtractorPieceUnit;
        tuning_.phase.store(ExtractorTuningPhase::kArmed);
    }

    /**
     * @brief Whether the in-situ tuning is armed or running, see tune().
     */
    bool tuning_pending() const noexcept
    {
        // Acquire, so that after the end of a tuning on another thread, which stores the index of
        // its choice before the phase, see record_(), algorithm_name() names that choice.
        return tuning_.phase.load(std::memory_order_acquire) != ExtractorTuningPhase::kOff;
    }

private:

    // The index of the candidate `C` in the plan: 1 + its position among the candidates, after the
    // estimate.
    template <typename C>
    static constexpr std::size_t candidate_index_ = [] {
        static_assert((std::same_as<C, Candidates> || ...), "Not a candidate of the plan.");
        std::size_t i = 1;
        (void) ((std::same_as<C, Candidates> ? false : (++i, true)) && ...);
        return i;
    }();

    // The index of the estimate's candidate, see visit_estimate().
    FISK_ALWAYS_INLINE static std::size_t estimate_index_()
    {
        return visit_estimate<Preference>(
            [](auto const& preferred) FISK_ALWAYS_INLINE_LAMBDA {
                return candidate_index_<std::remove_cvref_t<decltype(preferred)>>;
            }
        );
    }

    // The number of k-mers of a sequence, which start at its first `length - overlap` positions,
    // or 0 for a sequence shorter than a k-mer.
    std::size_t kmer_starts_(Seq const& seq) const noexcept
    {
        std::size_t length;
        if constexpr (std::same_as<Seq, std::string_view>) {
            length = seq.size();
        } else {
            length = seq.length();
        }
        std::size_t const overlap = extractor_overlap_(settings_);
        return length > overlap ? length - overlap : 0;
    }

    // The piece of `seq` with its `starts` k-mers from `at`, and the nucleotides that they reach.
    // For packed input a piece without a copy, see make_extractor_packed_piece(), also for all of
    // the sequence, so that the candidates run on one type of sequence.
    FISK_ALWAYS_INLINE auto piece_(
        Seq const& seq, std::size_t at, std::size_t starts
    ) const noexcept {
        std::size_t const overlap = extractor_overlap_(settings_);
        if constexpr (std::same_as<Seq, std::string_view>) {
            return seq.substr(at, starts + overlap);
        } else {
            return make_extractor_packed_piece(seq, at, starts, overlap);
        }
    }

    // Whether this call of for_each() tunes, while the in-situ tuning is armed or running: the
    // first thread that calls it after tune() takes the tuning over (kArmed to kClaiming,
    // then kRunning once it is set as the owner, which the other threads read only then), and
    // starts the turns with the estimate's candidate; the calls of other threads run the plan as it
    // is.
    bool claim_() const
    {
        auto phase = tuning_.phase.load(std::memory_order_acquire);
        if (phase == ExtractorTuningPhase::kArmed &&
            tuning_.phase.compare_exchange_strong(phase, ExtractorTuningPhase::kClaiming)) {
            tuning_.owner = std::this_thread::get_id();
            tuning_.start_turns(estimate_index_(), {Candidates::available()...});
            tuning_.phase.store(ExtractorTuningPhase::kRunning);
            return true;
        }
        return phase == ExtractorTuningPhase::kRunning &&
            tuning_.owner == std::this_thread::get_id();
    }

    // Count a piece of `starts` k-mers towards the turn; at the end of a turn, its time per k-mer;
    // after the last turn, keep the fastest candidate and end the tuning. Returns whether the
    // tuning goes on. Pieces without k-mers do not count.
    //
    // The time of a turn runs from its first piece, see for_each(), across the calls of
    // for_each() of its turn, so that it includes the code between them, the same for all
    // candidates. The clock is read only after every `piece_starts` k-mers, once per piece of a
    // long sequence or across shorter ones, as reading it orders the instructions around it: read
    // for every short sequence, such as a read, it would keep the processor from overlapping the
    // work of one sequence with that of the next, as it does without tuning, which slows down
    // some candidates more than others. A turn ends at the first reading after its time ran out,
    // and with a time of 0 or less after each piece.
    bool record_(std::size_t starts) const
    {
        auto& t = tuning_;
        if (starts == 0) {
            return true;
        }
        t.turn_starts += starts;
        t.unchecked_starts += starts;
        if (t.turn_time.count() > 0 && t.unchecked_starts < t.piece_starts) {
            return true;
        }
        t.unchecked_starts = 0;
        auto const elapsed = std::chrono::steady_clock::now() - t.turn_start;
        if (elapsed < t.turn_time) {
            return true;
        }
        std::size_t const c = t.order[t.turn % t.count] - 1;
        double const per_kmer = std::chrono::duration<double>(elapsed).count() /
            static_cast<double>(t.turn_starts);
        t.rate[c] = std::min(t.rate[c], per_kmer);
        t.turn_starts = 0;
        if (++t.turn < t.count * t.rounds) {
            return true;
        }
        std::size_t best = 0;
        for (std::size_t j = 1; j < t.rate.size(); ++j) {
            if (t.rate[j] < t.rate[best]) {
                best = j;
            }
        }
        index_.store(best + 1);
        t.phase.store(ExtractorTuningPhase::kOff);
        return false;
    }

    // The state of the in-situ tuning, see tune(). A copy of a plan whose tuning is armed
    // or running is armed, with the same settings, so that it tunes on its own.
    struct Tuning_
    {
        static constexpr std::size_t n = sizeof...(Candidates);

        CopyableAtomic<ExtractorTuningPhase> phase{ExtractorTuningPhase::kOff};
        std::thread::id owner{};

        // The settings, see tune().
        std::chrono::nanoseconds turn_time{};
        std::size_t rounds = 1;
        std::size_t piece_starts = kExtractorPieceUnit;

        // The indices of the available candidates in the order of the turns, and their number; the
        // turns so far; the start of the current one, its k-mers, and those since the clock was
        // last read; per candidate, by position, its fastest time per k-mer, in seconds.
        std::array<std::size_t, n> order{};
        std::size_t count = 0;
        std::size_t turn = 0;
        std::chrono::steady_clock::time_point turn_start{};
        std::size_t turn_starts = 0;
        std::size_t unchecked_starts = 0;
        std::array<double, n> rate{};

        Tuning_() = default;

        Tuning_(Tuning_ const& other) noexcept
            : phase(
                other.phase.load() == ExtractorTuningPhase::kOff
                    ? ExtractorTuningPhase::kOff : ExtractorTuningPhase::kArmed
            )
            , turn_time(other.turn_time)
            , rounds(other.rounds)
            , piece_starts(other.piece_starts)
        {}

        Tuning_& operator=(Tuning_ const& other) noexcept
        {
            Tuning_ const copy(other);
            phase.store(copy.phase.load());
            owner = {};
            turn_time = copy.turn_time;
            rounds = copy.rounds;
            piece_starts = copy.piece_starts;
            return *this;
        }

        // Start the turns: with the candidate of index `first`, then the other available ones.
        void start_turns(std::size_t first, std::array<bool, n> const& available) noexcept
        {
            count = 0;
            order[count++] = first;
            for (std::size_t j = 0; j < n; ++j) {
                if (available[j] && j + 1 != first) {
                    order[count++] = j + 1;
                }
            }
            turn = 0;
            turn_starts = 0;
            unchecked_starts = 0;
            rate.fill(std::numeric_limits<double>::infinity());
        }
    };

    Settings settings_;
    // The index of the candidate, written by the in-situ tuning in for_each(), hence mutable.
    mutable CopyableAtomic<std::size_t> index_{0};
    mutable Tuning_ tuning_;
};

} // namespace fisk
