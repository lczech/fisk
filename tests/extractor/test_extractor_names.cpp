#include <algorithm>
#include <cstddef>
#include <span>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/bit_extract/simd.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/extractor/internal.hpp"
#include "fisk/extractor/kmer_extractor.hpp"
#include "fisk/extractor/spaced_kmer_extractor.hpp"
#include "fisk/extractor/spaced_kmer_masks.hpp"
#include "fisk/kmer_extract/ascii_chunked_scalar.hpp"
#include "fisk/kmer_extract/ascii_chunked_simd.hpp"
#include "fisk/kmer_extract/packed_bytewise_scalar.hpp"
#include "fisk/kmer_extract/packed_bytewise_simd.hpp"
#include "fisk/spaced_kmer_extract/ascii_chunked_scalar.hpp"
#include "fisk/spaced_kmer_extract/ascii_chunked_simd.hpp"
#include "fisk/spaced_kmer_extract/packed_bytewise_scalar.hpp"
#include "fisk/spaced_kmer_extract/packed_bytewise_simd.hpp"
#include "extractor/common.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

constexpr auto kAcgt = Encoding::kACGT;
constexpr auto kMsb = Layout::kMSB;
constexpr auto kAscii = InputFormat::kAscii;
constexpr auto kPacked = InputFormat::kPacked;
constexpr auto kChecked = InputCheck::kChecked;
constexpr auto kUnchecked = InputCheck::kUnchecked;

// =================================================================================================
//     Algorithm Tables
// =================================================================================================

// The names of the algorithms of the extractors are strings, written by hand in their candidates.
// These tests tie each name to the function that it stands for: a table lists the algorithms of an
// extractor, each by a macro that takes the function once, and makes of it both the name, as a
// string, and a call. The compiler hence checks that each function of a table exists. The tests
// then check that the names of the extractor are exactly those of its table, and that the
// extractor, set to each name, gives the same k-mers as the function of that name.

// An algorithm of a table: its name, and a call of it with a sequence and a callback, as a lambda
// that takes the SIMD arch as a std::type_identity, for SIMD algorithms that stand for one per
// arch.
template <bool Simd, typename Call>
struct TestAlgorithm
{
    std::string name;
    Call call;
};

template <bool Simd, typename Call>
TestAlgorithm<Simd, Call> make_test_algorithm(std::string name, Call call)
{
    return { std::move(name), std::move(call) };
}

// The call of an algorithm, for the SIMD arch of the `simd` tag, which the scalar ones do not use.
#define FISK_TEST_ALGORITHM_CALL_(function, template_args, call_args)                         \
    [&](auto simd, auto const& seq, auto& callback) {                                          \
        using Simd [[maybe_unused]] = typename decltype(simd)::type;                           \
        function template_args call_args;                                                      \
    }

// An algorithm: `function`, called with the `template_args` and the `call_args`, which can use the
// `seq` and the `callback` of the test. Its name is the name of `function`, followed by the
// `suffix` of its template arguments, as the extractors name it.
#define FISK_TEST_ALGORITHM(function, template_args, suffix, call_args)                       \
    make_test_algorithm<false>(                                                                \
        std::string(#function) + suffix,                                                       \
        FISK_TEST_ALGORITHM_CALL_(function, template_args, call_args)                          \
    )

// A SIMD algorithm, as FISK_TEST_ALGORITHM, for each SIMD arch of the candidates: `Simd` in the
// `template_args` and `call_args` is the arch, whose name follows the `suffix` in the name.
#define FISK_TEST_SIMD_ALGORITHM(function, template_args, suffix, call_args)                  \
    make_test_algorithm<true>(                                                                 \
        std::string(#function) + suffix,                                                       \
        FISK_TEST_ALGORITHM_CALL_(function, template_args, call_args)                          \
    )

// Call `f(name, call)` for the `algorithm`, with the name and the call for the SIMD arch `Arch`.
template <typename Arch, typename Algorithm, typename F>
void call_test_algorithm(std::string const& name, Algorithm const& algorithm, F& f)
{
    f(name, [&](auto const& seq, auto& callback) {
        algorithm.call(std::type_identity<Arch>{}, seq, callback);
    });
}

// Call `f(name, call)` for each algorithm of the `table`, with the SIMD ones for each arch.
template <typename Table, typename F>
void for_each_test_algorithm(Table const& table, F&& f)
{
    auto const one = [&]<bool Simd, typename Call>(TestAlgorithm<Simd, Call> const& algorithm) {
        if constexpr (Simd) {
            [&]<typename... Arch>(TypeList<Arch...>) {
                (
                    call_test_algorithm<Arch>(
                        algorithm.name + "_" + simd_arch_name_<Arch>(), algorithm, f
                    ),
                    ...
                );
            }(CandidateSimdArchs{});
        } else {
            call_test_algorithm<SimdScalar>(algorithm.name, algorithm, f);
        }
    };
    std::apply([&](auto const&... algorithm) { (one(algorithm), ...); }, table);
}

// The names, sorted and joined, for comparing two lists of them.
inline std::string sorted_names(std::vector<std::string> names)
{
    std::sort(names.begin(), names.end());
    std::string result;
    for (auto const& name : names) {
        result += name + "\n";
    }
    return result;
}

// Check the names of the `extractor` against the `table`, and that each name runs its function.
// The tests are compiled for the CPU that they run on, so that all compiled algorithms are
// available, and the names of the extractor are all of its candidates.
template <typename Recorder, typename Extractor, typename Table>
void check_algorithm_table(Extractor extractor, Table const& table)
{
    // The names of the extractor are exactly those of the table.
    std::vector<std::string> names;
    for_each_test_algorithm(table, [&](std::string const& name, auto const&) {
        names.push_back(name);
    });
    ASSERT_EQ(sorted_names(extractor.algorithm_names()), sorted_names(names));

    // Set to each name, the extractor extracts the same as the function of that name.
    for_each_test_algorithm(table, [&](std::string const& name, auto const& call) {
        extractor.set_algorithm(name);
        for (auto const& seq : extractor_test_sequences<Extractor::input_check>()) {
            auto const input = extractor_test_input<Extractor>(seq);
            Recorder expected;
            call(input, expected);
            Recorder result;
            extractor.for_each(input, result);
            check_spaced_events(result.events, expected.events);
        }
    });
}

// =================================================================================================
//     K-mers
// =================================================================================================

constexpr std::size_t kK = 15;

// A callback that records the k-mers, as handed out one by one or in vectors, so that it serves
// the scalar and the SIMD functions alike.
struct KmerTableRecorder
{
    std::vector<SpacedEvent> events;

    template <typename K>
    void operator()(std::size_t pos, K const& kmer)
    {
        events.push_back({pos, 0, kmer_value(kmer)});
    }

    template <typename V, typename Valid>
    void operator()(std::size_t pos, V kmers, Valid valid)
    {
        for_each_valid_lane(pos, kmers, valid, *this);
    }
};

TEST(ExtractorNames, KmerAsciiChecked)
{
    ExtractorWordEncoder<kAcgt, kMsb> const encoder{};
    auto const table = std::tuple{
        FISK_TEST_ALGORITHM(
            for_each_kmer_ascii_checked_chunked_scalar, , "", (seq, kK, encoder, callback)
        ),
        FISK_TEST_SIMD_ALGORITHM(
            for_each_kmer_ascii_checked_chunked_simd, <Simd>, "", (seq, kK, encoder, callback)
        ),
    };
    check_algorithm_table<KmerTableRecorder>(
        KmerExtractor<kAscii, kChecked, kAcgt, kMsb>(kK), table
    );
}

TEST(ExtractorNames, KmerAsciiUnchecked)
{
    ExtractorWordEncoder<kAcgt, kMsb> const encoder{};
    auto const table = std::tuple{
        FISK_TEST_ALGORITHM(
            for_each_kmer_ascii_unchecked_chunked_scalar, , "", (seq, kK, encoder, callback)
        ),
        FISK_TEST_SIMD_ALGORITHM(
            for_each_kmer_ascii_unchecked_chunked_simd, <Simd>, "", (seq, kK, encoder, callback)
        ),
    };
    check_algorithm_table<KmerTableRecorder>(
        KmerExtractor<kAscii, kUnchecked, kAcgt, kMsb>(kK), table
    );
}

TEST(ExtractorNames, KmerPackedChecked)
{
    auto const table = std::tuple{
        FISK_TEST_ALGORITHM(
            for_each_kmer_packed_checked_bytewise_scalar, , "", (seq, kK, callback)
        ),
        FISK_TEST_SIMD_ALGORITHM(
            for_each_kmer_packed_checked_bytewise_simd, <Simd>, "", (seq, kK, callback)
        ),
    };
    check_algorithm_table<KmerTableRecorder>(
        KmerExtractor<kPacked, kChecked, kAcgt, kMsb>(kK), table
    );
}

TEST(ExtractorNames, KmerPackedUnchecked)
{
    auto const table = std::tuple{
        FISK_TEST_ALGORITHM(
            for_each_kmer_packed_unchecked_bytewise_scalar, , "", (seq, kK, callback)
        ),
        FISK_TEST_SIMD_ALGORITHM(
            for_each_kmer_packed_unchecked_bytewise_simd, <Simd>, "", (seq, kK, callback)
        ),
    };
    check_algorithm_table<KmerTableRecorder>(
        KmerExtractor<kPacked, kUnchecked, kAcgt, kMsb>(kK), table
    );
}

// =================================================================================================
//     Spaced K-mers
// =================================================================================================

constexpr std::size_t kTableMasks = 2;

using TableMasks = SpacedKmerMasks<kMsb, kTableMasks>;

// The masks in the form `BitExtractForm`, as the std::span of fixed size that the functions take.
template <typename BitExtractForm>
std::span<SpacedKmerMask<kMsb, BitExtractForm> const, kTableMasks> fixed_masks(
    TableMasks const& masks
) {
    return std::span<SpacedKmerMask<kMsb, BitExtractForm> const, kTableMasks>(
        masks.get<BitExtractForm>().data(), kTableMasks
    );
}

// A callback that records the spaced k-mers, as handed out one by one or in vectors, so that it
// serves the scalar and the SIMD functions alike.
struct SpacedTableRecorder
{
    std::vector<SpacedEvent> events;

    template <typename K>
    void operator()(std::size_t pos, std::size_t mask_idx, K const& kmer)
    {
        events.push_back({pos, mask_idx, kmer_value(kmer)});
    }

    template <typename V, typename Valid>
    void operator()(std::size_t pos, std::size_t mask_idx, V kmers, Valid valid)
    {
        for_each_valid_lane(pos, kmers, valid, [&](std::size_t lane_pos, auto const& kmer) {
            (*this)(lane_pos, mask_idx, kmer);
        });
    }
};

TableMasks const table_masks({"1101101", "1011011"});

TEST(ExtractorNames, SpacedAsciiChecked)
{
    ExtractorWordEncoder<kAcgt, kMsb> const encoder{};
    auto const& masks = table_masks;
    auto const table = std::tuple{
        #if defined(FISK_HAS_BMI2)
        FISK_TEST_ALGORITHM(
            for_each_spaced_kmer_ascii_checked_chunked_scalar, , "_pext",
            (seq, fixed_masks<BitExtractMask>(masks), encoder, bit_extract_pext, callback)
        ),
        #endif
        FISK_TEST_ALGORITHM(
            for_each_spaced_kmer_ascii_checked_chunked_scalar, , "_butterfly",
            (
                seq, fixed_masks<BitExtractButterflyMask>(masks), encoder, bit_extract_butterfly,
                callback
            )
        ),
        FISK_TEST_SIMD_ALGORITHM(
            for_each_spaced_kmer_ascii_checked_chunked_simd, , "_butterfly",
            (seq, fixed_masks<BitExtractKernelButterfly<Simd>>(masks), encoder, callback)
        ),
    };
    check_algorithm_table<SpacedTableRecorder>(
        SpacedKmerExtractor<kAscii, kChecked, kAcgt, kMsb, kTableMasks>(masks), table
    );
}

TEST(ExtractorNames, SpacedAsciiUnchecked)
{
    ExtractorWordEncoder<kAcgt, kMsb> const encoder{};
    auto const& masks = table_masks;
    auto const table = std::tuple{
        #if defined(FISK_HAS_BMI2)
        FISK_TEST_ALGORITHM(
            for_each_spaced_kmer_ascii_unchecked_chunked_scalar, , "_pext",
            (seq, fixed_masks<BitExtractMask>(masks), encoder, bit_extract_pext, callback)
        ),
        #endif
        FISK_TEST_ALGORITHM(
            for_each_spaced_kmer_ascii_unchecked_chunked_scalar, , "_butterfly",
            (
                seq, fixed_masks<BitExtractButterflyMask>(masks), encoder, bit_extract_butterfly,
                callback
            )
        ),
        FISK_TEST_SIMD_ALGORITHM(
            for_each_spaced_kmer_ascii_unchecked_chunked_simd, , "_butterfly",
            (seq, fixed_masks<BitExtractKernelButterfly<Simd>>(masks), encoder, callback)
        ),
    };
    check_algorithm_table<SpacedTableRecorder>(
        SpacedKmerExtractor<kAscii, kUnchecked, kAcgt, kMsb, kTableMasks>(masks), table
    );
}

TEST(ExtractorNames, SpacedPackedChecked)
{
    auto const& masks = table_masks;
    auto const table = std::tuple{
        #if defined(FISK_HAS_BMI2)
        FISK_TEST_ALGORITHM(
            for_each_spaced_kmer_packed_checked_bytewise_scalar, , "_pext",
            (seq, fixed_masks<BitExtractMask>(masks), bit_extract_pext, callback)
        ),
        FISK_TEST_ALGORITHM(
            for_each_spaced_kmer_packed_checked_bytewise_simd, , "_pext",
            (seq, fixed_masks<BitExtractKernelPext>(masks), callback)
        ),
        #endif
        FISK_TEST_ALGORITHM(
            for_each_spaced_kmer_packed_checked_bytewise_scalar, , "_butterfly",
            (seq, fixed_masks<BitExtractButterflyMask>(masks), bit_extract_butterfly, callback)
        ),
        FISK_TEST_SIMD_ALGORITHM(
            for_each_spaced_kmer_packed_checked_bytewise_simd, , "_butterfly",
            (seq, fixed_masks<BitExtractKernelButterfly<Simd>>(masks), callback)
        ),
    };
    check_algorithm_table<SpacedTableRecorder>(
        SpacedKmerExtractor<kPacked, kChecked, kAcgt, kMsb, kTableMasks>(masks), table
    );
}

TEST(ExtractorNames, SpacedPackedUnchecked)
{
    auto const& masks = table_masks;
    auto const table = std::tuple{
        #if defined(FISK_HAS_BMI2)
        FISK_TEST_ALGORITHM(
            for_each_spaced_kmer_packed_unchecked_bytewise_scalar, , "_pext",
            (seq, fixed_masks<BitExtractMask>(masks), bit_extract_pext, callback)
        ),
        FISK_TEST_ALGORITHM(
            for_each_spaced_kmer_packed_unchecked_bytewise_simd, , "_pext",
            (seq, fixed_masks<BitExtractKernelPext>(masks), callback)
        ),
        #endif
        FISK_TEST_ALGORITHM(
            for_each_spaced_kmer_packed_unchecked_bytewise_scalar, , "_butterfly",
            (seq, fixed_masks<BitExtractButterflyMask>(masks), bit_extract_butterfly, callback)
        ),
        FISK_TEST_SIMD_ALGORITHM(
            for_each_spaced_kmer_packed_unchecked_bytewise_simd, , "_butterfly",
            (seq, fixed_masks<BitExtractKernelButterfly<Simd>>(masks), callback)
        ),
    };
    check_algorithm_table<SpacedTableRecorder>(
        SpacedKmerExtractor<kPacked, kUnchecked, kAcgt, kMsb, kTableMasks>(masks), table
    );
}
