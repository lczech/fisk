#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fisk/util/random.hpp"
#include "sequence_pack_unchecked/bench.hpp"

namespace external_validation {

// =================================================================================================
//     Byte Comparison
// =================================================================================================

/**
 * @brief Build an independent scalar oracle for ACTG, including arbitrary input bytes.
 */
template <fisk::Layout L>
inline fisk::PackedSequence<fisk::Encoding::kACTG, L> expected_sequence(std::string_view seq)
{
    fisk::PackedSequence<fisk::Encoding::kACTG, L> expected(seq.size());
    auto const codes = expected.codes();
    for (std::size_t i = 0; i < seq.size(); ++i) {
        auto const c = static_cast<unsigned char>(seq[i]);
        std::uint8_t code;
        switch (c) {
            case 'A': case 'a': code = 0; break;
            case 'C': case 'c': code = 1; break;
            case 'T': case 't': code = 2; break;
            case 'G': case 'g': code = 3; break;
            default: code = static_cast<std::uint8_t>((c & 6) / 2); break;
        }
        std::size_t const shift = L == fisk::Layout::kMSB ? 6 - 2 * (i % 4) : 2 * (i % 4);
        codes[i / 4] |= static_cast<std::uint8_t>(code << shift);
    }
    return expected;
}

/**
 * @brief Compare every output byte and length, reusing dirty storage across calls.
 */
template <fisk::Layout L, typename Kernel>
inline void check_kernel(
    std::string_view name, Kernel kernel, std::vector<std::string> const& sequences
) {
    fisk::PackedSequence<fisk::Encoding::kACTG, L> out(512);
    std::fill(out.codes().begin(), out.codes().end(), std::uint8_t{0xFF});
    for (auto const& seq : sequences) {
        kernel(seq, out);
        auto const expected = expected_sequence<L>(seq);
        auto const out_codes = out.codes();
        auto const expected_codes = expected.codes();
        if (
            out.length() != expected.length() ||
            !std::equal(
                out_codes.begin(), out_codes.end(), expected_codes.begin(), expected_codes.end()
            )
        ) {
            throw std::runtime_error(
                std::string(name) + ": packed byte mismatch at input length "
                + std::to_string(seq.size())
            );
        }
    }
}

// =================================================================================================
//     Reference Validation
// =================================================================================================

/**
 * @brief Validate reference wrappers and fisk PEXT/scalar packing outside benchmark timing.
 */
inline void check_sequences(std::vector<std::string> const& sequences)
{
    using fisk::Layout;
    check_kernel<Layout::kMSB>("fastdna_lut", run_var_fastdna_lut, sequences);
    check_kernel<Layout::kMSB>(
        "fisk_direct_scalar_msb", run_var_unchecked_direct_scalar_actg_msb, sequences
    );
    check_kernel<Layout::kLSB>(
        "fisk_direct_scalar_lsb", run_var_unchecked_direct_scalar_actg_lsb, sequences
    );
    #if defined(FISK_HAS_BMI2)
    check_kernel<Layout::kMSB>("fastdna_pext_ordered", run_var_fastdna_pext_ordered, sequences);
    check_kernel<Layout::kLSB>("fastdna_pext_unordered", run_var_fastdna_pext_unordered, sequences);
    check_kernel<Layout::kLSB>(
        "fastdna_pext_unordered_unroll2", run_var_fastdna_pext_unordered_unroll2, sequences
    );
    check_kernel<Layout::kLSB>(
        "fastdna_pext_unordered_unroll4", run_var_fastdna_pext_unordered_unroll4, sequences
    );
    check_kernel<Layout::kMSB>("fisk_pext_msb", run_var_unchecked_pext_scalar_actg_msb, sequences);
    check_kernel<Layout::kLSB>("fisk_pext_lsb", run_var_unchecked_pext_scalar_actg_lsb, sequences);
    #endif
    #if defined(FISK_HAS_AVX2)
    check_kernel<Layout::kLSB>("cute_mul", run_var_cute_mul, sequences);
    check_kernel<Layout::kLSB>("cute_movemask", run_var_cute_movemask, sequences);
    #endif
}

/**
 * @brief Check tails, all input bytes, and the actual benchmark sequences before timing.
 */
inline void validate(std::vector<std::string> const& sequences)
{
    std::vector<std::string> corpus;
    fisk::Splitmix64 rng(3005);
    constexpr std::string_view alphabet = "ACGTNacgtn";
    for (std::size_t length = 0; length <= 129; ++length) {
        std::string seq(length, 'A');
        for (char& c : seq) {
            c = alphabet[rng.get_uint64() % alphabet.size()];
        }
        corpus.push_back(std::move(seq));
    }
    // Non-palindromic regressions for the original partial-byte and mixed-layout tails.
    for (auto const* seq : {"AC", "ACG", "GATTACCAACGTNacgt", "ACGTNacgtACGTNacgtACGTNacgtACGTNacgt"}) {
        corpus.emplace_back(seq);
    }
    for (unsigned int byte = 0; byte < 256; ++byte) {
        corpus.emplace_back(65, static_cast<char>(byte));
    }
    // End on shrinking outputs to expose stale bytes and padding.
    corpus.insert(corpus.end(), {"GG", "A", ""});
    check_sequences(corpus);
    check_sequences(sequences);
}

} // namespace external_validation
