# Example

[main.cpp](main.cpp) extracts all k-mers and spaced k-mers of the sequences of a FASTA file, such as a human genome, with the extractors of fisk: a `KmerExtractor` with k = 31, and a `SpacedKmerExtractor` with a mask of span 31 and weight 22. Both are checked: they skip the k-mers with characters other than ACGT, such as N. Each extractor chooses the fastest algorithm for its callback on this CPU, which counts the k-mers and sums them up as a checksum. The program prints, as CSV, per extractor: the chosen algorithm, the number of sequences, bases, and k-mers, the checksum, and the time to read the file and to extract the k-mers.

## Running it on a human genome

In the main directory of the repository,

```
example/run_human_genome.sh
```

downloads the human genome GRCh38 (the no-alt analysis set, 0.9 GB, from NCBI) into `example/data/`, unpacks it there (3.1 GB), builds the example in `example/build/`, and runs it. Later runs reuse the download; delete `example/data/` to remove it. `--genome t2t` uses T2T-CHM13v2.0 instead, and `--genome-file` a FASTA file of your own; `--help` lists all options.

## Building it

From the main directory of the repository, run:

```
cmake -S example -B build-example
cmake --build build-example
./build-example/fisk_example example/test.fa
```

[CMakeLists.txt](CMakeLists.txt) is that of a program that uses fisk: it adds fisk from the folder above, see [Using the library](../README.md#using-the-library) for other ways, and builds for the CPU of the computer it runs on.

## Notes

 - The callback takes one k-mer at a time, as most programs do. A callback that also takes the vectors of the SIMD algorithms can be faster, for consumers that work on vectors themselves, see [Tuned extraction](../README.md#tuned-extraction).
 - For input with only the characters ACGT, the unchecked extractors (`InputCheck::kUnchecked`) are faster.
 - The example uses a function per k-mer that does almost nothing, as a showcase of the pure extraction work. A program that does more with each k-mer of course takes additional time.
