// Standalone exploration benchmark for k-mers spanning 33--64 bases.
// Build with GCC and Clang using -O3 -march=native; this intentionally is not a
// library API.

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <immintrin.h>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/core/char_encoder.hpp"

namespace {

using u64 = std::uint64_t;
using u128 = unsigned __int128;

struct Wide2 {
  // Numeric limb order: words[0] is least significant.
  std::array<u64, 2> words{};

  bool operator==(Wide2 const &) const = default;
};

struct RunResult {
  u64 sink = 0;
  std::size_t count = 0;

  bool operator==(RunResult const &) const = default;
};

#if defined(__AVX2__)
struct alignas(32) ButterflyPlanAVX2 {
  __m256i mask{};
  __m256i sieves[6]{};
};
#endif

struct MaskPlan {
  std::string text;
  std::vector<unsigned> positions;
  Wide2 msb_mask;
  Wide2 lsb_mask;
  u64 position_mask_msb = 0;
  std::array<fisk::BitExtractButterflyTable, 2> msb_butterfly;
  std::array<fisk::BitExtractButterflyTable, 2> lsb_butterfly;
  std::array<fisk::BitExtractBlockTable, 2> msb_blocks;
  std::array<fisk::BitExtractBlockTable, 2> lsb_blocks;
#if defined(__AVX2__)
  std::array<ButterflyPlanAVX2, 2> msb_butterfly_avx2;
  std::array<ButterflyPlanAVX2, 2> lsb_butterfly_avx2;
#endif
  std::array<std::array<u64, 3>, 4> packed_masks{};
  std::array<std::array<unsigned, 3>, 4> packed_shifts{};
};

volatile u64 benchmark_sink = 0;

inline std::uint8_t encode(char c) noexcept {
  return fisk::CharEncoderTable<fisk::Encoding::kACGT>{}(c);
}

inline std::uint8_t encode_assume_valid(char c) noexcept {
  return fisk::CharEncoderAscii<fisk::Encoding::kACGT,
                                fisk::InputValidity::kAssumeValid>{}(c);
}

inline u64 high_mask(std::size_t k) noexcept {
  unsigned const high_bits = static_cast<unsigned>(2 * k - 64);
  return high_bits == 64 ? ~u64{0} : ((u64{1} << high_bits) - 1);
}

inline void shift_in(Wide2 &value, u64 code, u64 hi_mask) noexcept {
  value.words[1] = ((value.words[1] << 2) | (value.words[0] >> 62)) & hi_mask;
  value.words[0] = (value.words[0] << 2) | code;
}

inline bool contains(Wide2 value, Wide2 mask) noexcept {
  return (value.words[0] & mask.words[0]) == mask.words[0] &&
         (value.words[1] & mask.words[1]) == mask.words[1];
}

inline u64 combine_extracted(u64 lo, u64 hi, unsigned lo_bits) noexcept {
  return lo_bits == 64 ? lo : (lo | (hi << lo_bits));
}

inline u64 extract_pext(Wide2 value, Wide2 mask) noexcept {
  u64 const lo = _pext_u64(value.words[0], mask.words[0]);
  u64 const hi = _pext_u64(value.words[1], mask.words[1]);
  return combine_extracted(lo, hi, std::popcount(mask.words[0]));
}

inline u64 extract_butterfly(
    Wide2 value, Wide2 mask,
    std::array<fisk::BitExtractButterflyTable, 2> const &tables) noexcept {
  u64 const lo = fisk::bit_extract_butterfly_table(value.words[0], tables[0]);
  u64 const hi = fisk::bit_extract_butterfly_table(value.words[1], tables[1]);
  return combine_extracted(lo, hi, std::popcount(mask.words[0]));
}

inline u64 extract_blocks(
    Wide2 value, Wide2 mask,
    std::array<fisk::BitExtractBlockTable, 2> const &tables) noexcept {
  u64 const lo = fisk::bit_extract_block_table(value.words[0], tables[0]);
  u64 const hi = fisk::bit_extract_block_table(value.words[1], tables[1]);
  return combine_extracted(lo, hi, std::popcount(mask.words[0]));
}

#if defined(__AVX2__)
inline u64 horizontal_sum_avx2(__m256i value) noexcept {
  alignas(32) u64 lanes[4];
  _mm256_store_si256(reinterpret_cast<__m256i *>(lanes), value);
  return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}

inline __m256i extract_butterfly_avx2(__m256i value,
                                      ButterflyPlanAVX2 const &plan) noexcept {
  value = _mm256_and_si256(value, plan.mask);
  auto step = [&]<int Shift>(__m256i sieve) {
    __m256i const moved = _mm256_and_si256(value, sieve);
    value = _mm256_or_si256(_mm256_xor_si256(value, moved),
                            _mm256_srli_epi64(moved, Shift));
  };
  step.template operator()<1>(plan.sieves[0]);
  step.template operator()<2>(plan.sieves[1]);
  step.template operator()<4>(plan.sieves[2]);
  step.template operator()<8>(plan.sieves[3]);
  step.template operator()<16>(plan.sieves[4]);
  step.template operator()<32>(plan.sieves[5]);
  return value;
}
#endif

std::string make_mask(std::size_t span, std::size_t weight) {
  assert(span >= 33 && span <= 64);
  assert(weight >= 2 && weight <= std::min<std::size_t>(32, span));

  std::string mask(span, '0');
  // Even placement includes both endpoints and avoids accidental duplicates.
  for (std::size_t i = 0; i < weight; ++i) {
    std::size_t const pos = i * (span - 1) / (weight - 1);
    mask[pos] = '1';
  }
  // Integer division can collide only for weight > span, excluded above.
  assert(static_cast<std::size_t>(std::count(mask.begin(), mask.end(), '1')) ==
         weight);
  return mask;
}

MaskPlan prepare_mask(std::string text) {
  MaskPlan plan;
  plan.text = std::move(text);

  for (std::size_t i = 0; i < plan.text.size(); ++i) {
    bool const kept = plan.text[i] == '1';
    shift_in(plan.msb_mask, kept ? 3 : 0, ~u64{0});
    plan.position_mask_msb = (plan.position_mask_msb << 1) | kept;
    if (kept) {
      plan.positions.push_back(static_cast<unsigned>(i));
      unsigned const bit = static_cast<unsigned>(2 * i);
      plan.lsb_mask.words[bit / 64] |= u64{3} << (bit % 64);
    }
  }

  for (std::size_t limb = 0; limb < 2; ++limb) {
    plan.msb_butterfly[limb] =
        fisk::bit_extract_butterfly_table_preprocess(plan.msb_mask.words[limb]);
    plan.lsb_butterfly[limb] =
        fisk::bit_extract_butterfly_table_preprocess(plan.lsb_mask.words[limb]);
    plan.msb_blocks[limb] =
        fisk::bit_extract_block_table_preprocess(plan.msb_mask.words[limb]);
    plan.lsb_blocks[limb] =
        fisk::bit_extract_block_table_preprocess(plan.lsb_mask.words[limb]);
#if defined(__AVX2__)
    auto prepare_avx2 = [](fisk::BitExtractButterflyTable const &source,
                           ButterflyPlanAVX2 &target) {
      target.mask = _mm256_set1_epi64x(static_cast<long long>(source.mask));
      for (std::size_t i = 0; i < 6; ++i) {
        target.sieves[i] =
            _mm256_set1_epi64x(static_cast<long long>(source.sieves[i]));
      }
    };
    prepare_avx2(plan.msb_butterfly[limb], plan.msb_butterfly_avx2[limb]);
    prepare_avx2(plan.lsb_butterfly[limb], plan.lsb_butterfly_avx2[limb]);
#endif
  }

  // A packed LSB sequence has four possible two-bit starting offsets per byte.
  for (unsigned local = 0; local < 4; ++local) {
    unsigned const offset = 2 * local;
    auto &masks = plan.packed_masks[local];
    masks[0] = plan.lsb_mask.words[0] << offset;
    masks[1] = (plan.lsb_mask.words[1] << offset) |
               (offset == 0 ? 0 : plan.lsb_mask.words[0] >> (64 - offset));
    masks[2] = offset == 0 ? 0 : plan.lsb_mask.words[1] >> (64 - offset);

    auto &shifts = plan.packed_shifts[local];
    shifts[0] = 0;
    shifts[1] = std::popcount(masks[0]);
    shifts[2] = shifts[1] + std::popcount(masks[1]);
    assert(shifts[2] + std::popcount(masks[2]) == 2 * plan.positions.size());
  }
  return plan;
}

std::string make_sequence(std::size_t n, bool with_invalid) {
  std::mt19937_64 rng(0x6b6d6572);
  constexpr char bases[] = {'A', 'C', 'G', 'T'};
  std::string seq(n, 'A');
  for (std::size_t i = 0; i < n; ++i) {
    seq[i] = bases[rng() & 3u];
    if (with_invalid && i % 97 == 31) {
      seq[i] = 'N';
    }
  }
  return seq;
}

std::vector<std::uint8_t> pack_lsb(std::string_view seq) {
  // Padding permits the experimental grouped loaders to read three words at the
  // tail.
  std::vector<std::uint8_t> packed((seq.size() + 3) / 4 + 24, 0);
  for (std::size_t i = 0; i < seq.size(); ++i) {
    std::uint8_t const code = encode(seq[i]);
    assert(code < 4);
    packed[i / 4] |= static_cast<std::uint8_t>(code << (2 * (i % 4)));
  }
  return packed;
}

RunResult wide_array(std::string_view seq, std::size_t k) {
  Wide2 value;
  u64 const mask = high_mask(k);
  std::size_t valid = 0;
  RunResult result;
  for (char c : seq) {
    std::uint8_t const code = encode(c);
    shift_in(value, code & 3u, mask);
    valid = code < 4 ? valid + 1 : 0;
    if (valid >= k) {
      result.sink += value.words[0] + std::rotl(value.words[1], 17);
      ++result.count;
    }
  }
  return result;
}

RunResult wide_array_assume_valid(std::string_view seq, std::size_t k) {
  Wide2 value;
  u64 const mask = high_mask(k);
  RunResult result;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    shift_in(value, encode_assume_valid(seq[i]), mask);
    if (i + 1 >= k) {
      result.sink += value.words[0] + std::rotl(value.words[1], 17);
      ++result.count;
    }
  }
  return result;
}

RunResult wide_u128(std::string_view seq, std::size_t k) {
  u128 value = 0;
  u128 const mask = k == 64 ? ~u128{0} : ((u128{1} << (2 * k)) - 1);
  std::size_t valid = 0;
  RunResult result;
  for (char c : seq) {
    std::uint8_t const code = encode(c);
    value = ((value << 2) | (code & 3u)) & mask;
    valid = code < 4 ? valid + 1 : 0;
    if (valid >= k) {
      result.sink += static_cast<u64>(value) +
                     std::rotl(static_cast<u64>(value >> 64), 17);
      ++result.count;
    }
  }
  return result;
}

RunResult wide_vector(std::string_view seq, std::size_t k) {
  std::vector<u64> words(2);
  u64 const mask = high_mask(k);
  std::size_t valid = 0;
  RunResult result;
  for (char c : seq) {
    std::uint8_t const code = encode(c);
    u64 carry = code & 3u;
    for (u64 &word : words) {
      u64 const next = word >> 62;
      word = (word << 2) | carry;
      carry = next;
    }
    words[1] &= mask;
    valid = code < 4 ? valid + 1 : 0;
    if (valid >= k) {
      result.sink += words[0] + std::rotl(words[1], 17);
      ++result.count;
    }
  }
  return result;
}

RunResult spaced_array_pext(std::string_view seq, MaskPlan const &plan) {
  std::size_t const span = plan.text.size();
  u64 const mask = high_mask(span);
  Wide2 value;
  Wide2 valid;
  RunResult result;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    std::uint8_t const code = encode(seq[i]);
    shift_in(value, code & 3u, mask);
    shift_in(valid, code < 4 ? 3 : 0, mask);
    if (i + 1 >= span && contains(valid, plan.msb_mask)) {
      result.sink += extract_pext(value, plan.msb_mask);
      ++result.count;
    }
  }
  return result;
}

template <typename Extract>
RunResult spaced_array_extract(std::string_view seq, MaskPlan const &plan,
                               Extract &&extract) {
  std::size_t const span = plan.text.size();
  u64 const mask = high_mask(span);
  Wide2 value;
  Wide2 valid;
  RunResult result;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    std::uint8_t const code = encode(seq[i]);
    shift_in(value, code & 3u, mask);
    shift_in(valid, code < 4 ? 3 : 0, mask);
    if (i + 1 >= span && contains(valid, plan.msb_mask)) {
      result.sink += extract(value);
      ++result.count;
    }
  }
  return result;
}

RunResult spaced_array_butterfly(std::string_view seq, MaskPlan const &plan) {
  return spaced_array_extract(seq, plan, [&](Wide2 value) {
    return extract_butterfly(value, plan.msb_mask, plan.msb_butterfly);
  });
}

RunResult spaced_array_blocks(std::string_view seq, MaskPlan const &plan) {
  return spaced_array_extract(seq, plan, [&](Wide2 value) {
    return extract_blocks(value, plan.msb_mask, plan.msb_blocks);
  });
}

template <typename Extract>
RunResult spaced_array_extract_valid1(std::string_view seq,
                                      MaskPlan const &plan, Extract &&extract) {
  std::size_t const span = plan.text.size();
  u64 const mask = high_mask(span);
  Wide2 value;
  u64 valid = 0;
  RunResult result;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    std::uint8_t const code = encode(seq[i]);
    shift_in(value, code & 3u, mask);
    valid = (valid << 1) | static_cast<u64>(code < 4);
    if (i + 1 >= span &&
        (valid & plan.position_mask_msb) == plan.position_mask_msb) {
      result.sink += extract(value);
      ++result.count;
    }
  }
  return result;
}

RunResult spaced_array_pext_valid1(std::string_view seq, MaskPlan const &plan) {
  return spaced_array_extract_valid1(seq, plan, [&](Wide2 value) {
    return extract_pext(value, plan.msb_mask);
  });
}

RunResult spaced_array_butterfly_valid1(std::string_view seq,
                                        MaskPlan const &plan) {
  return spaced_array_extract_valid1(seq, plan, [&](Wide2 value) {
    return extract_butterfly(value, plan.msb_mask, plan.msb_butterfly);
  });
}

RunResult spaced_array_blocks_valid1(std::string_view seq,
                                     MaskPlan const &plan) {
  return spaced_array_extract_valid1(seq, plan, [&](Wide2 value) {
    return extract_blocks(value, plan.msb_mask, plan.msb_blocks);
  });
}

template <typename Extract>
RunResult spaced_array_extract_assume_valid(std::string_view seq,
                                            MaskPlan const &plan,
                                            Extract &&extract) {
  std::size_t const span = plan.text.size();
  u64 const mask = high_mask(span);
  Wide2 value;
  RunResult result;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    shift_in(value, encode_assume_valid(seq[i]), mask);
    if (i + 1 >= span) {
      result.sink += extract(value);
      ++result.count;
    }
  }
  return result;
}

RunResult spaced_array_butterfly_assume_valid(std::string_view seq,
                                              MaskPlan const &plan) {
  return spaced_array_extract_assume_valid(seq, plan, [&](Wide2 value) {
    return extract_butterfly(value, plan.msb_mask, plan.msb_butterfly);
  });
}

RunResult spaced_array_pext_assume_valid(std::string_view seq,
                                         MaskPlan const &plan) {
  return spaced_array_extract_assume_valid(seq, plan, [&](Wide2 value) {
    return extract_pext(value, plan.msb_mask);
  });
}

RunResult spaced_array_pext_assume_valid_unroll4(std::string_view seq,
                                                 MaskPlan const &plan) {
  std::size_t const span = plan.text.size();
  if (seq.size() < span) {
    return {};
  }
  u64 const mask = high_mask(span);
  Wide2 value;
  std::size_t i = 0;
  for (; i + 1 < span; ++i) {
    shift_in(value, encode_assume_valid(seq[i]), mask);
  }

  RunResult result;
  for (; i + 4 <= seq.size(); i += 4) {
    u64 const packed1 = encode_assume_valid(seq[i]);
    u64 const packed2 = (packed1 << 2) | encode_assume_valid(seq[i + 1]);
    u64 const packed3 = (packed2 << 2) | encode_assume_valid(seq[i + 2]);
    u64 const packed4 = (packed3 << 2) | encode_assume_valid(seq[i + 3]);
    u64 const low = value.words[0];
    u64 const high = value.words[1];
    Wide2 const value0{
        {(low << 2) | packed1, ((high << 2) | (low >> 62)) & mask}};
    Wide2 const value1{
        {(low << 4) | packed2, ((high << 4) | (low >> 60)) & mask}};
    Wide2 const value2{
        {(low << 6) | packed3, ((high << 6) | (low >> 58)) & mask}};
    Wide2 const value3{
        {(low << 8) | packed4, ((high << 8) | (low >> 56)) & mask}};
    result.sink += extract_pext(value0, plan.msb_mask) +
                   extract_pext(value1, plan.msb_mask) +
                   extract_pext(value2, plan.msb_mask) +
                   extract_pext(value3, plan.msb_mask);
    result.count += 4;
    value = value3;
  }
  for (; i < seq.size(); ++i) {
    shift_in(value, encode_assume_valid(seq[i]), mask);
    result.sink += extract_pext(value, plan.msb_mask);
    ++result.count;
  }
  return result;
}

#if defined(__AVX2__)
template <bool VectorSink>
RunResult spaced_array_butterfly_assume_valid_avx2_impl(std::string_view seq,
                                                        MaskPlan const &plan) {
  std::size_t const span = plan.text.size();
  if (seq.size() < span) {
    return {};
  }
  u64 const mask = high_mask(span);
  Wide2 value;
  std::size_t i = 0;
  for (; i + 1 < span; ++i) {
    shift_in(value, encode_assume_valid(seq[i]), mask);
  }

  __m256i const output_shift =
      _mm256_set1_epi64x(std::popcount(plan.msb_mask.words[0]));
  // Model a vector-capable consumer without a scalar handoff per batch.
  __m256i vector_sum = _mm256_setzero_si256();
  RunResult result;
  for (; i + 4 <= seq.size(); i += 4) {
    u64 const packed1 = encode_assume_valid(seq[i]);
    u64 const packed2 = (packed1 << 2) | encode_assume_valid(seq[i + 1]);
    u64 const packed3 = (packed2 << 2) | encode_assume_valid(seq[i + 2]);
    u64 const packed4 = (packed3 << 2) | encode_assume_valid(seq[i + 3]);
    u64 const low = value.words[0];
    u64 const high = value.words[1];
    u64 const low0 = (low << 2) | packed1;
    u64 const low1 = (low << 4) | packed2;
    u64 const low2 = (low << 6) | packed3;
    u64 const low3 = (low << 8) | packed4;
    u64 const high0 = ((high << 2) | (low >> 62)) & mask;
    u64 const high1 = ((high << 4) | (low >> 60)) & mask;
    u64 const high2 = ((high << 6) | (low >> 58)) & mask;
    u64 const high3 = ((high << 8) | (low >> 56)) & mask;
    __m256i const low_values = _mm256_set_epi64x(low3, low2, low1, low0);
    __m256i const high_values = _mm256_set_epi64x(high3, high2, high1, high0);
    __m256i const extracted_low =
        extract_butterfly_avx2(low_values, plan.msb_butterfly_avx2[0]);
    __m256i const extracted_high =
        extract_butterfly_avx2(high_values, plan.msb_butterfly_avx2[1]);
    __m256i const output = _mm256_or_si256(
        extracted_low, _mm256_sllv_epi64(extracted_high, output_shift));
    if constexpr (VectorSink) {
      vector_sum = _mm256_add_epi64(vector_sum, output);
    } else {
      result.sink += horizontal_sum_avx2(output);
    }
    result.count += 4;
    value = Wide2{{low3, high3}};
  }
  for (; i < seq.size(); ++i) {
    shift_in(value, encode_assume_valid(seq[i]), mask);
    result.sink += extract_butterfly(value, plan.msb_mask, plan.msb_butterfly);
    ++result.count;
  }
  if constexpr (VectorSink) {
    result.sink += horizontal_sum_avx2(vector_sum);
  }
  return result;
}

RunResult spaced_array_butterfly_assume_valid_avx2(std::string_view seq,
                                                   MaskPlan const &plan) {
  return spaced_array_butterfly_assume_valid_avx2_impl<false>(seq, plan);
}

RunResult
spaced_array_butterfly_assume_valid_avx2_vector(std::string_view seq,
                                                MaskPlan const &plan) {
  return spaced_array_butterfly_assume_valid_avx2_impl<true>(seq, plan);
}
#endif

RunResult spaced_u128_pext(std::string_view seq, MaskPlan const &plan) {
  std::size_t const span = plan.text.size();
  u128 const width_mask = span == 64 ? ~u128{0} : ((u128{1} << (2 * span)) - 1);
  u128 const mask = static_cast<u128>(plan.msb_mask.words[0]) |
                    (static_cast<u128>(plan.msb_mask.words[1]) << 64);
  u128 value = 0;
  u128 valid = 0;
  RunResult result;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    std::uint8_t const code = encode(seq[i]);
    value = ((value << 2) | (code & 3u)) & width_mask;
    valid = ((valid << 2) | (code < 4 ? 3 : 0)) & width_mask;
    if (i + 1 >= span && (valid & mask) == mask) {
      Wide2 const wide{
          {static_cast<u64>(value), static_cast<u64>(value >> 64)}};
      result.sink += extract_pext(wide, plan.msb_mask);
      ++result.count;
    }
  }
  return result;
}

RunResult spaced_direct_ascii(std::string_view seq, MaskPlan const &plan) {
  RunResult result;
  if (seq.size() < plan.text.size()) {
    return result;
  }
  std::size_t const stop = seq.size() - plan.text.size();
  for (std::size_t start = 0; start <= stop; ++start) {
    u64 out = 0;
    bool valid = true;
    for (unsigned pos : plan.positions) {
      std::uint8_t const code = encode(seq[start + pos]);
      valid &= code < 4;
      out = (out << 2) | (code & 3u);
    }
    if (valid) {
      result.sink += out;
      ++result.count;
    }
  }
  return result;
}

inline u64 extract_positions_msb(Wide2 value, MaskPlan const &plan) noexcept {
  u64 out = 0;
  std::size_t const span = plan.text.size();
  for (unsigned pos : plan.positions) {
    unsigned const bit = static_cast<unsigned>(2 * (span - 1 - pos));
    out = (out << 2) | ((value.words[bit / 64] >> (bit % 64)) & 3u);
  }
  return out;
}

template <typename Extract>
RunResult extract_many(std::span<Wide2 const> values, Extract &&extract) {
  RunResult result;
  for (Wide2 value : values) {
    result.sink += extract(value);
    ++result.count;
  }
  return result;
}

std::vector<Wide2> make_wide_values(std::size_t count, std::size_t span) {
  std::mt19937_64 rng(0x77696465 + span);
  std::vector<Wide2> values(count);
  for (Wide2 &value : values) {
    value.words = {rng(), rng() & high_mask(span)};
  }
  return values;
}

inline std::array<u64, 3>
load_packed_words(std::span<std::uint8_t const> packed, std::size_t byte) {
  std::array<u64, 3> words;
  std::memcpy(&words[0], packed.data() + byte, 8);
  std::memcpy(&words[1], packed.data() + byte + 8, 8);
  std::memcpy(&words[2], packed.data() + byte + 16, 8);
  return words;
}

inline u64
extract_packed_pext_skip_zero(std::array<u64, 3> const &words,
                              std::array<u64, 3> const &masks,
                              std::array<unsigned, 3> const &shifts) noexcept {
  u64 out = _pext_u64(words[0], masks[0]) |
            (_pext_u64(words[1], masks[1]) << shifts[1]);
  // Local offset zero never reaches word 2; avoid both PEXT and a shift by 64.
  if (masks[2] != 0) {
    out |= _pext_u64(words[2], masks[2]) << shifts[2];
  }
  return out;
}

inline u64
extract_packed_pext_two(std::array<u64, 3> const &words,
                        std::array<u64, 3> const &masks,
                        std::array<unsigned, 3> const &shifts) noexcept {
  return _pext_u64(words[0], masks[0]) |
         (_pext_u64(words[1], masks[1]) << shifts[1]);
}

RunResult spaced_direct_packed_looped(std::span<std::uint8_t const> packed,
                                      std::size_t seq_size,
                                      MaskPlan const &plan) {
  RunResult result;
  std::size_t const stop = seq_size - plan.text.size();
  for (std::size_t byte = 0; 4 * byte <= stop; ++byte) {
    auto const words = load_packed_words(packed, byte);
    unsigned const last_local =
        static_cast<unsigned>(std::min<std::size_t>(3, stop - 4 * byte));
    for (unsigned local = 0; local <= last_local; ++local) {
      result.sink += extract_packed_pext_skip_zero(
          words, plan.packed_masks[local], plan.packed_shifts[local]);
      ++result.count;
    }
  }
  return result;
}

RunResult spaced_direct_packed_unrolled(std::span<std::uint8_t const> packed,
                                        std::size_t seq_size,
                                        MaskPlan const &plan) {
  RunResult result;
  std::size_t const window_count = seq_size - plan.text.size() + 1;
  std::size_t const full_bytes = window_count / 4;
  for (std::size_t byte = 0; byte < full_bytes; ++byte) {
    auto const words = load_packed_words(packed, byte);
    result.sink += extract_packed_pext_two(words, plan.packed_masks[0],
                                           plan.packed_shifts[0]);
    result.sink += extract_packed_pext_skip_zero(words, plan.packed_masks[1],
                                                 plan.packed_shifts[1]);
    result.sink += extract_packed_pext_skip_zero(words, plan.packed_masks[2],
                                                 plan.packed_shifts[2]);
    result.sink += extract_packed_pext_skip_zero(words, plan.packed_masks[3],
                                                 plan.packed_shifts[3]);
    result.count += 4;
  }

  if (std::size_t const remaining = window_count % 4; remaining != 0) {
    auto const words = load_packed_words(packed, full_bytes);
    result.sink += extract_packed_pext_two(words, plan.packed_masks[0],
                                           plan.packed_shifts[0]);
    ++result.count;
    for (std::size_t local = 1; local < remaining; ++local) {
      result.sink += extract_packed_pext_skip_zero(
          words, plan.packed_masks[local], plan.packed_shifts[local]);
      ++result.count;
    }
  }
  return result;
}

RunResult spaced_window_packed(std::span<std::uint8_t const> packed,
                               std::size_t seq_size, MaskPlan const &plan) {
  RunResult result;
  std::size_t const stop = seq_size - plan.text.size();
  for (std::size_t byte = 0; 4 * byte <= stop; ++byte) {
    auto const words = load_packed_words(packed, byte);
    unsigned const last_local =
        static_cast<unsigned>(std::min<std::size_t>(3, stop - 4 * byte));
    for (unsigned local = 0; local <= last_local; ++local) {
      unsigned const offset = 2 * local;
      Wide2 window;
      if (offset == 0) {
        window = Wide2{{words[0], words[1]}};
      } else {
        window.words[0] = (words[0] >> offset) | (words[1] << (64 - offset));
        window.words[1] = (words[1] >> offset) | (words[2] << (64 - offset));
      }
      result.sink += extract_pext(window, plan.lsb_mask);
      ++result.count;
    }
  }
  return result;
}

template <typename Extract>
RunResult spaced_window_packed_extract(std::span<std::uint8_t const> packed,
                                       std::size_t seq_size,
                                       MaskPlan const &plan,
                                       Extract &&extract) {
  RunResult result;
  std::size_t const stop = seq_size - plan.text.size();
  for (std::size_t byte = 0; 4 * byte <= stop; ++byte) {
    auto const words = load_packed_words(packed, byte);
    unsigned const last_local =
        static_cast<unsigned>(std::min<std::size_t>(3, stop - 4 * byte));
    for (unsigned local = 0; local <= last_local; ++local) {
      unsigned const offset = 2 * local;
      Wide2 window;
      if (offset == 0) {
        window = Wide2{{words[0], words[1]}};
      } else {
        window.words[0] = (words[0] >> offset) | (words[1] << (64 - offset));
        window.words[1] = (words[1] >> offset) | (words[2] << (64 - offset));
      }
      result.sink += extract(window);
      ++result.count;
    }
  }
  return result;
}

RunResult spaced_window_packed_butterfly(std::span<std::uint8_t const> packed,
                                         std::size_t seq_size,
                                         MaskPlan const &plan) {
  return spaced_window_packed_extract(packed, seq_size, plan, [&](Wide2 value) {
    return extract_butterfly(value, plan.lsb_mask, plan.lsb_butterfly);
  });
}

#if defined(__AVX2__)
template <bool VectorSink>
RunResult
spaced_window_packed_butterfly_avx2_impl(std::span<std::uint8_t const> packed,
                                         std::size_t seq_size,
                                         MaskPlan const &plan) {
  RunResult result;
  std::size_t const stop = seq_size - plan.text.size();
  __m256i const right = _mm256_set_epi64x(6, 4, 2, 0);
  __m256i const left = _mm256_set_epi64x(58, 60, 62, 64);
  __m256i const output_shift =
      _mm256_set1_epi64x(std::popcount(plan.lsb_mask.words[0]));
  // Model a vector-capable consumer without a scalar handoff per batch.
  __m256i vector_sum = _mm256_setzero_si256();

  for (std::size_t byte = 0; 4 * byte <= stop; ++byte) {
    auto const words = load_packed_words(packed, byte);
    __m256i const word0 = _mm256_set1_epi64x(words[0]);
    __m256i const word1 = _mm256_set1_epi64x(words[1]);
    __m256i const word2 = _mm256_set1_epi64x(words[2]);
    __m256i const low_windows = _mm256_or_si256(_mm256_srlv_epi64(word0, right),
                                                _mm256_sllv_epi64(word1, left));
    __m256i const high_windows = _mm256_or_si256(
        _mm256_srlv_epi64(word1, right), _mm256_sllv_epi64(word2, left));
    __m256i const low =
        extract_butterfly_avx2(low_windows, plan.lsb_butterfly_avx2[0]);
    __m256i const high =
        extract_butterfly_avx2(high_windows, plan.lsb_butterfly_avx2[1]);
    __m256i const output =
        _mm256_or_si256(low, _mm256_sllv_epi64(high, output_shift));

    unsigned const count =
        static_cast<unsigned>(std::min<std::size_t>(4, stop + 1 - 4 * byte));
    if constexpr (VectorSink) {
      if (count == 4) {
        vector_sum = _mm256_add_epi64(vector_sum, output);
      } else {
        alignas(32) u64 lanes[4];
        _mm256_store_si256(reinterpret_cast<__m256i *>(lanes), output);
        for (unsigned lane = 0; lane < count; ++lane) {
          result.sink += lanes[lane];
        }
      }
    } else {
      alignas(32) u64 lanes[4];
      _mm256_store_si256(reinterpret_cast<__m256i *>(lanes), output);
      for (unsigned lane = 0; lane < count; ++lane) {
        result.sink += lanes[lane];
      }
    }
    result.count += count;
  }
  if constexpr (VectorSink) {
    result.sink += horizontal_sum_avx2(vector_sum);
  }
  return result;
}

RunResult
spaced_window_packed_butterfly_avx2(std::span<std::uint8_t const> packed,
                                    std::size_t seq_size,
                                    MaskPlan const &plan) {
  return spaced_window_packed_butterfly_avx2_impl<false>(packed, seq_size,
                                                         plan);
}

RunResult
spaced_window_packed_butterfly_avx2_vector(std::span<std::uint8_t const> packed,
                                           std::size_t seq_size,
                                           MaskPlan const &plan) {
  return spaced_window_packed_butterfly_avx2_impl<true>(packed, seq_size, plan);
}
#endif

RunResult spaced_window_packed_blocks(std::span<std::uint8_t const> packed,
                                      std::size_t seq_size,
                                      MaskPlan const &plan) {
  return spaced_window_packed_extract(packed, seq_size, plan, [&](Wide2 value) {
    return extract_blocks(value, plan.lsb_mask, plan.lsb_blocks);
  });
}

RunResult spaced_positions_packed_fixed(std::span<std::uint8_t const> packed,
                                        std::size_t seq_size,
                                        MaskPlan const &plan) {
  RunResult result;
  std::size_t const stop = seq_size - plan.text.size();
  for (std::size_t start = 0; start <= stop; ++start) {
    u64 out = 0;
    for (std::size_t lane = 0; lane < plan.positions.size(); ++lane) {
      std::size_t const index = start + plan.positions[lane];
      u64 const code = (packed[index / 4] >> (2 * (index % 4))) & 3u;
      out |= code << (2 * lane);
    }
    result.sink += out;
    ++result.count;
  }
  return result;
}

template <std::size_t Words>
RunResult generic_array_rolling(std::string_view seq, std::size_t k) {
  std::array<u64, Words> words{};
  unsigned const used_high_bits =
      static_cast<unsigned>(2 * k - 64 * (Words - 1));
  u64 const top_mask =
      used_high_bits == 64 ? ~u64{0} : ((u64{1} << used_high_bits) - 1);
  std::size_t valid = 0;
  RunResult result;
  for (char c : seq) {
    std::uint8_t const code = encode(c);
    u64 carry = code & 3u;
    for (u64 &word : words) {
      u64 const next = word >> 62;
      word = (word << 2) | carry;
      carry = next;
    }
    words.back() &= top_mask;
    valid = code < 4 ? valid + 1 : 0;
    if (valid >= k) {
      result.sink += words[0];
      ++result.count;
    }
  }
  return result;
}

RunResult scalar_rolling(std::string_view seq, std::size_t k) {
  u64 const mask = k == 32 ? ~u64{0} : ((u64{1} << (2 * k)) - 1);
  u64 value = 0;
  std::size_t valid = 0;
  RunResult result;
  for (char c : seq) {
    std::uint8_t const code = encode(c);
    value = ((value << 2) | (code & 3u)) & mask;
    valid = code < 4 ? valid + 1 : 0;
    if (valid >= k) {
      result.sink += value;
      ++result.count;
    }
  }
  return result;
}

template <typename Function>
double benchmark(Function &&function, RunResult expected, int repeats = 7) {
  double best = std::numeric_limits<double>::infinity();
  for (int repeat = 0; repeat < repeats; ++repeat) {
    auto const start = std::chrono::steady_clock::now();
    RunResult const got = function();
    auto const stop = std::chrono::steady_clock::now();
    if (got != expected) {
      std::cerr << "correctness failure: got sink=" << got.sink
                << " count=" << got.count << ", expected sink=" << expected.sink
                << " count=" << expected.count << '\n';
      std::exit(2);
    }
    benchmark_sink = benchmark_sink ^ got.sink;
    double const ns =
        std::chrono::duration<double, std::nano>(stop - start).count();
    best = std::min(best, ns);
  }
  return best / static_cast<double>(expected.count);
}

void print_row(std::string_view suite, std::size_t span, std::size_t weight,
               std::string_view variant, double ns_per_output) {
  std::cout << suite << ',' << span << ',' << weight << ',' << variant << ','
            << std::fixed << std::setprecision(4) << ns_per_output << '\n';
}

void self_test() {
  std::string const seq = make_sequence(400, true);
  for (std::size_t span : {33u, 48u, 64u}) {
    RunResult const wide_expected = wide_array(seq, span);
    assert(wide_u128(seq, span) == wide_expected);
    assert(wide_vector(seq, span) == wide_expected);
    for (std::size_t weight : {4u, 16u, 32u}) {
      MaskPlan const plan = prepare_mask(make_mask(span, weight));
      RunResult const expected = spaced_direct_ascii(seq, plan);
      assert(spaced_array_pext(seq, plan) == expected);
      assert(spaced_array_butterfly(seq, plan) == expected);
      assert(spaced_array_blocks(seq, plan) == expected);
      assert(spaced_array_pext_valid1(seq, plan) == expected);
      assert(spaced_array_butterfly_valid1(seq, plan) == expected);
      assert(spaced_array_blocks_valid1(seq, plan) == expected);
      assert(spaced_u128_pext(seq, plan) == expected);

      auto const values = make_wide_values(100, span);
      auto const extract_expected = extract_many(values, [&](Wide2 value) {
        return extract_positions_msb(value, plan);
      });
      assert(extract_many(values, [&](Wide2 value) {
               return extract_pext(value, plan.msb_mask);
             }) == extract_expected);
      assert(extract_many(values, [&](Wide2 value) {
               return extract_butterfly(value, plan.msb_mask,
                                        plan.msb_butterfly);
             }) == extract_expected);
      assert(extract_many(values, [&](Wide2 value) {
               return extract_blocks(value, plan.msb_mask, plan.msb_blocks);
             }) == extract_expected);

      std::string const valid_seq = make_sequence(400, false);
      RunResult const valid_expected = spaced_direct_ascii(valid_seq, plan);
      assert(wide_array_assume_valid(valid_seq, span) ==
             wide_array(valid_seq, span));
      assert(spaced_array_butterfly_assume_valid(valid_seq, plan) ==
             valid_expected);
      assert(spaced_array_pext_assume_valid(valid_seq, plan) == valid_expected);
      assert(spaced_array_pext_assume_valid_unroll4(valid_seq, plan) ==
             valid_expected);
#if defined(__AVX2__)
      assert(spaced_array_butterfly_assume_valid_avx2(valid_seq, plan) ==
             valid_expected);
      assert(spaced_array_butterfly_assume_valid_avx2_vector(valid_seq, plan) ==
             valid_expected);
#endif
      auto const packed = pack_lsb(valid_seq);
      RunResult const packed_expected =
          spaced_positions_packed_fixed(packed, valid_seq.size(), plan);
      assert(spaced_direct_packed_looped(packed, valid_seq.size(), plan) ==
             packed_expected);
      assert(spaced_direct_packed_unrolled(packed, valid_seq.size(), plan) ==
             packed_expected);
      assert(spaced_window_packed(packed, valid_seq.size(), plan) ==
             packed_expected);
      assert(spaced_window_packed_butterfly(packed, valid_seq.size(), plan) ==
             packed_expected);
#if defined(__AVX2__)
      assert(spaced_window_packed_butterfly_avx2(packed, valid_seq.size(),
                                                 plan) == packed_expected);
      assert(spaced_window_packed_butterfly_avx2_vector(
                 packed, valid_seq.size(), plan) == packed_expected);
#endif
      assert(spaced_window_packed_blocks(packed, valid_seq.size(), plan) ==
             packed_expected);
    }
  }
  assert(generic_array_rolling<1>(seq, 31) == scalar_rolling(seq, 31));
}

} // namespace

int main() {
  self_test();
  std::size_t constexpr sequence_size = 1u << 20;
  std::string const ascii = make_sequence(sequence_size, false);
  std::string const valid_ascii = ascii;
  auto const packed = pack_lsb(valid_ascii);

  std::cout << "suite,span,weight,variant,ns_per_output\n";

  for (std::size_t span : {33u, 48u, 64u}) {
    auto const wide_values = make_wide_values(sequence_size, span);
    RunResult const expected = wide_array(ascii, span);
    print_row("wide_ascii", span, 0, "array2",
              benchmark([&] { return wide_array(ascii, span); }, expected));
    print_row("wide_ascii", span, 0, "array2_assume_valid",
              benchmark([&] { return wide_array_assume_valid(ascii, span); },
                        expected));
    print_row("wide_ascii", span, 0, "uint128",
              benchmark([&] { return wide_u128(ascii, span); }, expected));
    print_row("wide_ascii", span, 0, "vector2",
              benchmark([&] { return wide_vector(ascii, span); }, expected));

    for (std::size_t weight : {4u, 16u, 32u}) {
      MaskPlan const plan = prepare_mask(make_mask(span, weight));
      RunResult const extract_expected =
          extract_many(wide_values, [&](Wide2 value) {
            return extract_positions_msb(value, plan);
          });
      print_row("wide_to_spaced", span, weight, "pext2",
                benchmark(
                    [&] {
                      return extract_many(wide_values, [&](Wide2 value) {
                        return extract_pext(value, plan.msb_mask);
                      });
                    },
                    extract_expected));
      print_row("wide_to_spaced", span, weight, "butterfly2",
                benchmark(
                    [&] {
                      return extract_many(wide_values, [&](Wide2 value) {
                        return extract_butterfly(value, plan.msb_mask,
                                                 plan.msb_butterfly);
                      });
                    },
                    extract_expected));
      print_row("wide_to_spaced", span, weight, "blocks2",
                benchmark(
                    [&] {
                      return extract_many(wide_values, [&](Wide2 value) {
                        return extract_blocks(value, plan.msb_mask,
                                              plan.msb_blocks);
                      });
                    },
                    extract_expected));
      print_row("wide_to_spaced", span, weight, "positions",
                benchmark(
                    [&] {
                      return extract_many(wide_values, [&](Wide2 value) {
                        return extract_positions_msb(value, plan);
                      });
                    },
                    extract_expected));

      RunResult const spaced_expected = spaced_direct_ascii(ascii, plan);
      print_row("spaced_ascii", span, weight, "array2_pext",
                benchmark([&] { return spaced_array_pext(ascii, plan); },
                          spaced_expected));
      print_row("spaced_ascii", span, weight, "array2_butterfly",
                benchmark([&] { return spaced_array_butterfly(ascii, plan); },
                          spaced_expected));
      print_row("spaced_ascii", span, weight, "array2_blocks",
                benchmark([&] { return spaced_array_blocks(ascii, plan); },
                          spaced_expected));
      print_row("spaced_ascii", span, weight, "array2_pext_valid1",
                benchmark([&] { return spaced_array_pext_valid1(ascii, plan); },
                          spaced_expected));
      print_row(
          "spaced_ascii", span, weight, "array2_butterfly_valid1",
          benchmark([&] { return spaced_array_butterfly_valid1(ascii, plan); },
                    spaced_expected));
      print_row(
          "spaced_ascii", span, weight, "array2_blocks_valid1",
          benchmark([&] { return spaced_array_blocks_valid1(ascii, plan); },
                    spaced_expected));
      print_row(
          "spaced_ascii", span, weight, "array2_butterfly_assume_valid",
          benchmark(
              [&] { return spaced_array_butterfly_assume_valid(ascii, plan); },
              spaced_expected));
      print_row(
          "spaced_ascii", span, weight, "array2_pext_assume_valid",
          benchmark([&] { return spaced_array_pext_assume_valid(ascii, plan); },
                    spaced_expected));
      print_row(
          "spaced_ascii", span, weight, "array2_pext_assume_valid_unroll4",
          benchmark(
              [&] {
                return spaced_array_pext_assume_valid_unroll4(ascii, plan);
              },
              spaced_expected));
#if defined(__AVX2__)
      print_row(
          "spaced_ascii", span, weight, "array2_butterfly_assume_valid_avx2",
          benchmark(
              [&] {
                return spaced_array_butterfly_assume_valid_avx2(ascii, plan);
              },
              spaced_expected));
      print_row("spaced_ascii", span, weight,
                "array2_butterfly_assume_valid_avx2_vector",
                benchmark(
                    [&] {
                      return spaced_array_butterfly_assume_valid_avx2_vector(
                          ascii, plan);
                    },
                    spaced_expected));
#endif
      print_row("spaced_ascii", span, weight, "uint128_pext",
                benchmark([&] { return spaced_u128_pext(ascii, plan); },
                          spaced_expected));
      print_row("spaced_ascii", span, weight, "direct_positions",
                benchmark([&] { return spaced_direct_ascii(ascii, plan); },
                          spaced_expected));

      RunResult const packed_expected =
          spaced_positions_packed_fixed(packed, valid_ascii.size(), plan);
      print_row("spaced_packed", span, weight, "direct_pext_looped",
                benchmark(
                    [&] {
                      return spaced_direct_packed_looped(
                          packed, valid_ascii.size(), plan);
                    },
                    packed_expected));
      print_row("spaced_packed", span, weight, "direct_pext_unrolled",
                benchmark(
                    [&] {
                      return spaced_direct_packed_unrolled(
                          packed, valid_ascii.size(), plan);
                    },
                    packed_expected));
      print_row("spaced_packed", span, weight, "window_pext2",
                benchmark(
                    [&] {
                      return spaced_window_packed(packed, valid_ascii.size(),
                                                  plan);
                    },
                    packed_expected));
      print_row("spaced_packed", span, weight, "window_butterfly2",
                benchmark(
                    [&] {
                      return spaced_window_packed_butterfly(
                          packed, valid_ascii.size(), plan);
                    },
                    packed_expected));
#if defined(__AVX2__)
      print_row("spaced_packed", span, weight, "window_butterfly2_avx2",
                benchmark(
                    [&] {
                      return spaced_window_packed_butterfly_avx2(
                          packed, valid_ascii.size(), plan);
                    },
                    packed_expected));
      print_row("spaced_packed", span, weight, "window_butterfly2_avx2_vector",
                benchmark(
                    [&] {
                      return spaced_window_packed_butterfly_avx2_vector(
                          packed, valid_ascii.size(), plan);
                    },
                    packed_expected));
#endif
      print_row("spaced_packed", span, weight, "window_blocks2",
                benchmark(
                    [&] {
                      return spaced_window_packed_blocks(
                          packed, valid_ascii.size(), plan);
                    },
                    packed_expected));
      print_row("spaced_packed", span, weight, "direct_positions",
                benchmark(
                    [&] {
                      return spaced_positions_packed_fixed(
                          packed, valid_ascii.size(), plan);
                    },
                    packed_expected));
    }
  }

  RunResult const narrow_expected = scalar_rolling(ascii, 31);
  print_row(
      "narrow_unification", 31, 0, "scalar",
      benchmark([&] { return scalar_rolling(ascii, 31); }, narrow_expected));
  print_row("narrow_unification", 31, 0, "array1_generic",
            benchmark([&] { return generic_array_rolling<1>(ascii, 31); },
                      narrow_expected));

  return benchmark_sink == 0x9e3779b97f4a7c15ull ? 1 : 0;
}
