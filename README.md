# GPU Roofline Profiler

A CUDA profiling suite for my RTX 5070. It measures this machine's peak compute
throughput and memory bandwidth (both theoretical and empirically measured),
places a ladder of hand written kernels on a roofline plot against those
ceilings, and explains each kernel's position with real Nsight Compute counters,
all compiled into one LaTeX PDF report.

The point is not another GEMM. The point is the ladder from SAXPY to a register
blocked GEMM walking the arithmetic intensity axis, with a counter backed reason
for where each kernel lands relative to the roofline, and an honest accounting of
the gap between the vendor spec sheet and what the card actually delivers.

## Status

Early build. Phase 0 (environment detection and scaffold) is complete. The
kernel suite, harness, profiling, analysis, and report are being built in the
order laid out in the roadmap. Runtime estimates in the spec are placeholders and
will be replaced with measured wall clock once the timing sweep runs.

## Target machine

| Component | Spec |
|---|---|
| CPU | Intel Core i7-14700K |
| RAM | 32 GB DDR5 |
| GPU | NVIDIA GeForce RTX 5070, 12 GB GDDR7, Blackwell (GB205), sm_120 |
| OS | Windows 11 Pro |

All performance numbers in this repo come from runs on this GPU. Nothing from a
spec sheet is ever presented as measured.

## Quickstart

To be filled in and verified once the pipeline runs end to end.

## Repository layout

```
src/         CUDA kernels, profiling utilities, benchmark driver
configs/     sweep parameters (YAML); changing a sweep never touches source
tests/       GoogleTest correctness tests, run via ctest
python/      roofline analysis package, CLI, and its pytest suite
scripts/     pipeline, nsys/ncu wrappers, style checks
docs/        architecture, methodology, kernel notes, profiling guide, logs
report/      main LaTeX report
report_debug/  engineering postmortem report
report_for_me/ standalone personal documentation report
results/     committed sample dataset plus the compiled report PDF
```

## Known limitations

To be filled in honestly as they surface. CI has no GPU, so it compiles the
`.cu` files and runs the CPU side tests only; GPU correctness and performance
are not CI testable and this is disclosed in the workflow.

## License

MIT. See [LICENSE](LICENSE).
