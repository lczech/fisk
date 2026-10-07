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
only be compared within the same file.
