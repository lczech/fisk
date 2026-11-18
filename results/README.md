# Results

The results of the [benchmarks](../benchmarks/README.md), for each CPU and compiler that we tested, as written by `scripts/run_benchmarks.sh`. The plots are not in git; to make them, see [plot/README.md](../plot/README.md).

The results of one CPU and compiler are in `<CPU>/<Compiler>_<version>/`, such as `Intel_Core_7_240H/GCC_15/`:

 - `<mode>/`, for each mode of handling the results (`sum`, `barrier`, `write`, see [Measurements](../benchmarks/README.md#measurements)): one CSV per suite, `sys_info.txt` with the system, the build, and the benchmark settings, and `log.txt` with the output; `genome_sweep.csv`, with its own `genome_sweep_sys_info.txt` and `genome_sweep_log.txt`.
 - `human_genome.csv`: the output of the [example](../example/README.md) on the human genome.
 - `DuoHash.csv`: the results of the DuoHash benchmark, see below, where we ran it.

The columns of the CSVs of the suites are described in [Measurements](../benchmarks/README.md#measurements).

**DuoHash.** The [DuoHash](https://github.com/CominLab/DuoHash) repository has a benchmark program of the spaced k-mer hashing methods FSH, ISSH, and MISSH, which [our fork of it](https://github.com/lczech/DuoHash) runs for the mask sets W10L15 to W26L31, for one mask (`single`) and for several (`multi`). It is run separately from `scripts/run_benchmarks.sh`, see the fork. Its `DuoHash.csv` has the time per operation of each method and case, as `ns_per_op`.
