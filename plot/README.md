# Plots

The scripts that make the plots of the benchmark results, from the CSV files in `results/`, as written by [scripts/run_benchmarks.sh](../scripts/run_benchmarks.sh). The plots are not in git; run the scripts to make them.

## Requirements

Python with the packages of `conda-env.yaml`, available as `python`, for instance in its conda environment:

```
conda env create -f plot/conda-env.yaml
conda activate fisk-plot
```

`inkscape` converts the SVG plots to PDF; without it, there are only the PNG and SVG plots.

## Usage

```
./plot/plot_all.sh
```

makes all plots, in three steps, which can also be run on their own:

 - `plot_run.sh [<run folder>]`: the plots of one benchmark run, such as `results/Intel_Core_7_240H/GCC_15/sum/`, into that folder. Without a folder, those of every run in `results/`.
 - `plot_summaries.sh <sum|barrier|write>`: the plots across CPUs and compilers, of one sink mode, into `results/Summaries/`. The CPUs and compilers to show are in its `CPUS` list. Next to each bar plot, a CSV table gives the fastest implementation of each of them.
 - `collect.sh --all`: copies of the plots of all runs, grouped by plot, into `results/Collect/<plot>/<CPU>_<Compiler>.png`, so that one plot can be compared across them.

The plots are named `<plot>_<sink>_<unit>_<scale>`, such as `kmer_extract_ascii_supp_sum_ops_log.png`: the sink mode of the results, and the unit (`ops` for throughput, `ns` for time per operation) and scale (`log` or `linear`) of the y-axis; the combinations to make are in the `COMBOS` lists of the scripts. Many plots come in several levels of detail, `main`, `supp`, and `full`, which are part of `<plot>`.
