# Wide k-mer experiment

Standalone prototype for spans 33--64. It compares fixed and dynamic storage, extraction of a
64-bit spaced k-mer from a two-word k-mer, and fused extraction from ASCII and packed sequence
data. This is exploratory code, not a proposed public API.

Build and run with either compiler:

```sh
mkdir -p build
g++ -std=c++20 -O3 -march=native -Iinclude -Wall -Wextra -Werror \
    experiments/wide_kmers.cpp -o build/wide_kmers_gcc
taskset -c 2 ./build/wide_kmers_gcc

clang++ -std=c++20 -O3 -march=native -Iinclude -Wall -Wextra -Werror \
    experiments/wide_kmers.cpp -o build/wide_kmers_clang
taskset -c 2 ./build/wide_kmers_clang
```

The checked-in CSVs were produced on an AMD Ryzen 7 PRO 4750U. Times are best-of-seven and should
only be compared within the same file. The `*_novec.csv` files repeat the experiment with GCC's
`-fno-tree-vectorize` or Clang's `-fno-vectorize -fno-slp-vectorize`; they bound how much the sum
sink benefits from batched compiler code generation. The `*_valid1.csv` files compare the original
two-bit validity state with the optimized one-bit-per-base state in the same run. The `*_avx2.csv`
and `*_avx2_novec.csv` files compare scalar packed-window butterfly extraction with an explicit
four-window AVX2 implementation. The `*_assume.csv` files measure an assume-valid ASCII path and
also retain the rejected direct-position/block candidates for comparison. The `*_roll4.csv` and
`*_roll4_novec.csv` files measure four-at-once rolling plus explicit AVX2 extraction; they also
retain the rejected scalar-only unrolling candidate.
The `*_immediate.csv` and `*_immediate_novec.csv` files replace AVX2 variable-count shifts with
compile-time immediate shifts.
The `*_vector_sink.csv` and `*_vector_sink_novec.csv` files keep four extracted outputs in an AVX2
register for a vector-capable consumer instead of reducing every group to scalar values.
The `wide_kmers_intel_*_baseline.csv` files repeat the final baseline on an Intel Core 7 240H,
pinned to performance core 4, using GCC 15.2 and Clang 21.1.
The `*_pext_assume.csv` files add an assume-valid two-limb PEXT path; `wide_kmers_intel_*` names
are Gaia results, while names without `intel` are from the AMD Zen 2 host.
