# Methodology

How the numbers are measured and counted, so a reader can trust or challenge
them.

## Timing

- CUDA events, never host wall clock. Device time only.
- Many launches between one event pair, then divide by the launch count, so per
  launch overhead does not dominate the cheap kernels.
- Warmup iterations discarded first, to let the boost clock ramp and caches warm.
- At least ten timed batches; mean, median, and standard deviation reported. This
  is a boost clocked consumer card that also drives the display, so variance is
  real and is shown rather than smoothed away.

## Counting floating point operations

From each kernel's actual arithmetic, not an estimate:

| kernel | FLOPs |
|---|---|
| SAXPY over N elements | 2N |
| GEMV, M by N | 2MN |
| GEMM, M by N by K | 2MNK |
| transpose | 0 |

A multiply and an add count as two operations; a fused multiply add is two.

## Counting bytes

Two counts, reported together where both exist:

- **Theoretical**: the minimum a perfect implementation moves, reading each input
  once and writing each output once in the kernel's precision.
- **Measured**: DRAM read and write bytes from Nsight Compute, which include
  cache misses, spills, and redundant loads a real kernel suffers.

Arithmetic intensity uses measured bytes where the profiler ran and falls back to
theoretical bytes elsewhere, always labeled. The gap between the two intensities
is the point of the analysis, not a rounding detail.

## Theoretical versus empirical peak

- **Theoretical compute**: SM count times FP32 lanes per SM times boost clock,
  times two for the fused multiply add, all queried from `cudaGetDeviceProperties`
  at runtime.
- **Theoretical bandwidth**: bus width times the effective memory transfer rate.
  GDDR7 quotes an effective rate that already folds in its signalling, so the
  derivation avoids double counting the data rate; which convention is used is
  recorded next to the code.
- **Empirical compute**: a register resident FMA loop pushed as hard as the card
  allows.
- **Empirical bandwidth**: a large device to device copy.

Both ceilings appear on every roofline, visually distinct. A card almost never
reaches its theoretical peak, and showing the measured ceiling next to it is what
keeps the plot honest.

## Tolerances

Correctness is checked against a CPU reference at small sizes and against cuBLAS
at large sizes. GPU floating point accumulation order differs from a sequential
CPU sum, so an exact match is the wrong expectation. The large GEMM check uses a
relative error tolerance on the order of 1e-4, which is normal for fp32
accumulation over thousands of terms; the exact value and its justification live
in a comment next to each check.
