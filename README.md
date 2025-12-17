# fisk
FISK: Fast Iteration of Spaced K-mers

## Build and execute

Simply call

```
make
```

to build the program, and

```
taskset -c 2 ./bin/fisk
```

to execute it (includine thread pinning for optimized performance).
