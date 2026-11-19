# Example

[main.cpp](main.cpp) extracts all k-mers and spaced k-mers of the sequences of a FASTA file, such as a human genome, with `for_each_kmer()` (k = 31) and `for_each_spaced_kmer()` (a mask of span 31 and weight 22). Both are checked: they skip the k-mers with characters other than ACGT, such as N. Each runs the algorithm that is likely the fastest on this CPU, with a callback that counts the k-mers and sums them up as a checksum. The program prints, as CSV, per extraction: the algorithm, the number of sequences, nucleotides, and k-mers, the checksum, and the time to read the file and to extract the k-mers.

## Running it on a human genome

In the main directory of the repository,

```
example/run_human_genome.sh
```

downloads the human genome GRCh38 (the no-alt analysis set, 0.9 GB, from NCBI) into `example/data/`, unpacks it there (3.1 GB), builds the example in `example/build/<compiler>/`, and runs it. This needs CMake (3.20 or later), a C++20 compiler, curl, and about 5 GB of disk space and 4 GB of memory. Later runs reuse the download; delete `example/data/` to remove it. `--genome t2t` uses T2T-CHM13v2.0 instead, and `--genome-file` a FASTA file of your own; `--help` lists all options.

## Building it

From the main directory of the repository, run:

```
cmake -S example -B build-example
cmake --build build-example
./build-example/fisk_example example/test.fa
```

[CMakeLists.txt](CMakeLists.txt) is that of a program that uses fisk: it adds fisk from the folder above, see [Quick start](../README.md#quick-start) for other ways, and builds for the CPU of the computer it runs on.

## Notes

 - The callback takes one k-mer at a time, as most programs do. A callback that also takes the vectors of the SIMD algorithms can be faster, for consumers that work on vectors themselves, see [Details of the extraction](../README.md#details-of-the-extraction).
 - For input with only the characters ACGT, the unchecked extraction (`InputCheck::kUnchecked`) is faster.
 - The example reads the sequences into `AlignedString`, and builds with code alignment flags, for the best speed; any string works, and the flags are optional, see [Performance and build options](../README.md#performance-and-build-options).
 - The example uses a function per k-mer that does almost nothing, as a showcase of the pure extraction work. A program that does more with each k-mer of course takes additional time.
