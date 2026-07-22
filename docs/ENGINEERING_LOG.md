# Engineering log

Dated entries written at the moment a problem or decision happens: symptom,
root cause, options weighed, chosen fix and why, commit, verification. This
file is the raw material for the debug report (Section 14), so I keep it honest
and I never pad it.

## 2026-07-22 Phase 0: environment detection

**Symptom.** Fresh clone on the target machine. Ran the Section 3 tool probes
before writing any build code.

**What I found.**

Present and working:

- GPU: NVIDIA GeForce RTX 5070, 12 GB, driver 610.62, CUDA UMD version 13.3,
  reported by `nvidia-smi`. Idle at 36 C, 25 W, so thermal headroom is fine.
- `cmake` 4.3.1, `python` 3.12.11, `git` 2.53.0, MiKTeX with `pdflatex`
  (MiKTeX-pdfTeX 4.24, MiKTeX 26.1).

Missing, and each one is a hard blocker for a different phase:

- **CUDA Toolkit / `nvcc`.** Not installed anywhere. No
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`, no `CUDA_PATH`. Only the
  display driver and runtime (UMD) are present. Without `nvcc` no `.cu` file
  compiles, so Phases 1 and 2 cannot start.
- **MSVC / `cl.exe`.** No Visual Studio and no Build Tools. `vswhere.exe` is
  absent. On Windows `nvcc` shells out to `cl.exe` as its host compiler, so this
  blocks compilation just as hard as a missing `nvcc`.
- **Nsight Systems (`nsys`) and Nsight Compute (`ncu`).** Neither installed.
  These block Phase 4 profiling, but not earlier phases.
- **`perl`.** MiKTeX ships `latexmk` as a Perl script and there is no Perl on the
  machine, so `latexmk` errors out immediately. `pdflatex` itself runs, so the
  report can still compile through a direct `pdflatex` plus `bibtex` sequence if
  I decide not to depend on `latexmk`.
- **`ninja`.** Absent. Not fatal; CMake can drive the MSVC generator instead,
  but Ninja is nicer and cheap to add.

**Root cause.** The machine has the gaming and display stack (driver, NVIDIA
App, telemetry) but none of the developer toolchain. Nothing is broken; the
tools were simply never installed.

**Decision required.** The CUDA Toolkit, MSVC Build Tools, and the Nsight pair
are large downloads that need elevated permissions to install. Spec Section 0
and Section 3 both say to stop and ask before installing at this fork rather
than run installers unattended. Raised the choice with the repo owner. Fix and
verification recorded in the next entry once the route is chosen.

**Done without the toolchain in the meantime.** Initialized the git repo,
created the full directory layout (Section 6), wrote the dash lint
(`scripts/check_no_dashes.py`) so the style gate is live from Phase 0, captured
the environment snapshot to `results/sample_run/phase0_environment.txt`, and
started this log and `DESIGN_DECISIONS.md`. None of that depends on a compiler.

## 2026-07-22 The dash lint tripped over itself

**Symptom.** First run of `check_no_dashes.py` reported three violations: one in
the upstream spec document and two in the lint script itself, on the very lines
that define the characters it searches for.

**Root cause.** The detection constants were written as literal em and en dash
characters, so the scanner found its own needles. Obvious in hindsight.

**Fix.** Build the constants from code points with `chr(0x2014)` and
`chr(0x2013)`, so the file contains no literal dash. Exempted the upstream spec
document by name: it is an external input, not an authored deliverable, and I am
not going to edit the document I was handed. Verified clean afterward.

## 2026-07-22 Python pins had no wheels for this interpreter

**Symptom.** `pip install -r requirements.txt` died building pandas from source:
`ERROR: Could not find vswhere.exe` out of meson.

**Root cause.** Two layers. The machine's usable Python is CPython 3.14 (the
system `python` is an msys2 build with no pip at all; the real one is behind the
`py` launcher). The spec's suggested pins are older releases that ship no 3.14
wheels, so pip fell back to a source build, which needs the same MSVC that is
missing for `nvcc`.

**Options.** Install an older Python alongside, or move the pins forward to
releases that have 3.14 wheels.

**Fix and why.** Moved the pins forward and re-pinned `requirements.txt` to
exactly the versions that installed and passed here (numpy 2.5.1, pandas 3.0.3,
matplotlib 3.11.1, pytest 9.1.1, ruff 0.15.22, pyyaml 6.0.3, tqdm 4.69.0).
Pinning to what actually builds on this machine is more honest and more
reproducible than pinning to an aspiration that cannot be built. Adding a second
Python would have been extra surface area for no gain.

**Verification.** Fresh venv, clean install, 34 tests pass, ruff clean.

## 2026-07-22 NVML timestamp parsing dropped a valid row

**Symptom.** `test_load_nvml_ok` failed: a two row sample where the first
timestamp had no fractional seconds and the second had one tenth of a second
parsed the first row and turned the second into `NaT`, so the loader rejected a
file that was actually valid.

**Root cause.** The loader called `pd.to_datetime` with no format, letting pandas
infer per element. Pandas warned it was falling back to `dateutil` per element,
and under pandas 3.0 the mixed precision made that inference inconsistent.

**Fix.** Pin the parse to `format="ISO8601"`, which accepts both precisions and
still rejects genuinely malformed input loudly. The NVML monitor I write later
emits ISO 8601, so this is a constraint I control rather than a guess.

**Verification.** Test passes, and the warning is gone.

## 2026-07-22 Report placeholder broke the LaTeX compile

**Symptom.** `pdflatex` fatal error at the first guarded table:
`Missing $ inserted` on `\inputgenerated{tables/environment_summary.tex}`.

**Root cause.** The guard macro prints the missing artifact's path inside
`\texttt` when the artifact does not exist. Those paths contain underscores, and
LaTeX reads a text mode underscore as a math subscript.

**Fix.** Wrap the path in `\detokenize` so the underscores render literally. I
preferred this over renaming the generated files, because underscored file names
are conventional and clear, and the placeholder is the thing that should adapt.

**Verification.** All three reports compile: `main.pdf`, `debug_report.pdf`, and
`report_for_me.pdf`.

## 2026-07-22 Build tooling gaps worked around, not papered over

Two smaller environment facts worth recording. `make` is not on PATH, though
msys2 ships one at `C:\msys64\usr\bin\make.exe`. And MiKTeX's `latexmk` cannot
run at all without a Perl engine. Rather than make the report build depend on
either, the report Makefiles try `latexmk` first and fall back to a direct
`pdflatex` plus `bibtex` sequence, and `scripts/build_reports.ps1` does the same
thing with no `make` and no Perl. That script is the one I actually verified end
to end. Once Strawberry Perl is installed the `latexmk` path takes over and the
fallback stops being reached.

## 2026-07-22 Toolchain installed, and CUDA 13 removed the clock properties

**Context.** Toolchain now present: CUDA 13.3.73, MSVC 14.44.35207 (Build Tools
2022), Nsight Compute 2026.2.1, Nsight Systems 2026.1.3, Ninja 1.12.0, Perl
5.42.2.

**First symptom.** A trivial smoke test would not compile:

```text
error: class "cudaDeviceProp" has no member "clockRate"
error: class "cudaDeviceProp" has no member "memoryClockRate"
```

**Root cause.** CUDA 13 removed `clockRate` and `memoryClockRate` from
`cudaDeviceProp`. They were deprecated through 12.x and are now gone. This
matters well beyond a smoke test: the theoretical peak derivation needs both the
SM clock and the memory clock, so the whole ceiling calculation depended on two
fields that no longer exist.

**Fix.** Query them through `cudaDeviceGetAttribute` with `cudaDevAttrClockRate`
and `cudaDevAttrMemoryClockRate`, which are still supported and returned status 0
on this card. Both are read at runtime and their status is checked, so if a future
toolkit removes these too the failure is loud rather than a silent zero that
would quietly produce a peak of zero and a nonsense roofline.

**Verification, and why I trust the derivation.** The queried values are 48 SMs,
2.625 GHz boost, 14.001 GHz memory clock, 192 bit bus. Deriving from those:

- FP32 peak: 48 SMs times 128 lanes is 6144 CUDA cores, which matches the vendor
  reference exactly. Times 2 for the fused multiply add, times 2.625 GHz, gives
  32.3 TFLOP/s against a vendor quote of about 31 TFLOP/s at a lower reference
  boost clock. Consistent.
- Bandwidth: 192 bits is 24 bytes, times 14.001 GHz, times 2 for the double data
  rate, gives 672.0 GB/s against a vendor quote of 672 GB/s. Exact.

Both derivations landing on the reference numbers is the check that the formulas
are right. A derived peak far from the reference would have been a bug in my
arithmetic, not a hardware surprise, exactly as the spec warns.

## 2026-07-22 CMake targeted the wrong GPU architecture

**Symptom.** The first successful configure printed `CUDA architectures: 75`.
The card is sm_120.

**Root cause.** I set `CMAKE_CUDA_ARCHITECTURES` inside an
`if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)` guard placed after `project()`.
Enabling the CUDA language in `project()` installs a default value, so by the
time my guard ran the variable was already defined and my auto detection never
executed.

**Why it mattered.** This is the quiet, expensive kind of bug. Nothing fails; the
build just silently compiles for an older architecture than the card, and every
performance number that followed would have been measured on code generated for
the wrong target.

**Fix.** Choose the architecture before `project()`. Order of preference is an
explicit `CUDAARCHS`, then `native` to detect the installed GPU, then 120 as the
fallback for a GPU free CI runner. Verified: configure now reports `native`.

## 2026-07-22 yaml-cpp does not configure under CMake 4

**Symptom.** `FetchContent` pulled yaml-cpp 0.8.0 and configure failed with
"Compatibility with CMake < 3.5 has been removed from CMake."

**Root cause.** This machine has CMake 4.3.1, which dropped support for the very
old `cmake_minimum_required` values that yaml-cpp 0.8.0 still declares.

**Options.** Vendor a patched copy, drop the dependency and hand write a YAML
subset parser, or tell CMake to treat the old declaration as 3.5.

**Fix and why.** Set `CMAKE_POLICY_VERSION_MINIMUM` to 3.5 immediately around the
`FetchContent_MakeAvailable` call and unset it afterward, so the compatibility
shim applies to the fetched dependency and not to my own project. Hand writing a
YAML parser to dodge one upstream version declaration would have been a poor
trade, and vendoring a patched copy is a maintenance burden for the same result.

## 2026-07-22 Nsight Compute collected nothing, convincingly

**Symptom.** The first `ncu` run looked like a success. It attached to the
process, reported the kernel by name with its grid and block dimensions, ran the
kernel, and disconnected cleanly. The metric value was `n/a`.

**Why this is the dangerous failure mode.** Nothing errored. A CSV full of `n/a`
parses fine, and had I piped it straight into the analysis without looking, the
result would have been a report whose counter columns were all empty for a reason
no one had noticed.

**Root cause.** On Windows the NVIDIA driver restricts GPU performance counters
to administrators unless
`HKLM:\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak\RmProfilingAdminOnly`
is set to 0. I checked: the value is not set at all, so the driver default
applies, and the shell was not elevated. This is precisely the counter access
caveat the spec warns about for WSL2, and it turns out to bite on native Windows
too.

**Fix.** `scripts/run_ncu_profile.ps1` now checks for elevation up front and
refuses with an explanation rather than producing a file of `n/a`. The pass is
run once from an elevated shell. I did not set the registry value: it is a
permanent, system wide relaxation of who may read GPU counters, and that is the
machine owner's decision to make deliberately, not a side effect of me wanting a
profile.

**Verified separately.** Nsight Systems does *not* need elevation. It warns that
CPU sampling and context switch tracing are disabled, neither of which this
project uses, and produces a complete CUDA and NVTX timeline regardless. So the
timeline half of Phase 4 was finished without waiting on anything.

## 2026-07-22 NVTX broke the build through windows.h

**Symptom.** Enabling NVTX turned a clean build into
`timing.hpp(139): error C2059: syntax error: ')'`, in a file that had not
changed and that has nothing to do with profiling.

**Root cause.** `nvtx3/nvToolsExt.h` includes `windows.h`, which defines `min`
and `max` as preprocessor macros. Line 139 of the timing harness is
`std::max(1, std::min(wanted, max_launches))`, and once those are macros the
expression stops parsing. The error points at the victim, not the culprit.

**Fix.** Define `NOMINMAX` (and `WIN32_LEAN_AND_MEAN`) in the NVTX helper before
it includes the header, so the macros are never created. Putting the guard next
to the include that causes the problem keeps the fix where the next person will
look, rather than burying it in a global compile flag.

**Also fixed alongside.** The CMake NVTX detection used
`EXISTS "${CUDAToolkit_INCLUDE_DIRS}/nvtx3/nvToolsExt.h"`, which silently never
matched because that variable is a *list* of two directories, not one path. It
reported "NVTX not found" on a machine where the header was plainly present.
Replaced with `find_path`, which searches the list properly.

## 2026-07-22 NVML explains the ceiling that looked impossible

**Symptom.** The measured FP32 peak came out at 33.54 TFLOP/s against a
theoretical 32.26, or 104 percent of a number nothing is supposed to exceed.

**Root cause, from the NVML trace.** The sweep's NVML log records an SM clock
averaging 2857 MHz and peaking at 2865 MHz under load, while
`cudaDevAttrClockRate` reports 2625 MHz. The runtime reports a nominal boost
clock; the card actually boosts past it. Recomputing the ceiling at the observed
2865 MHz gives 6144 lanes times 2 times 2.865 GHz, or 35.2 TFLOP/s, and the
measured 33.54 is then 95 percent of it, which is exactly where a good FMA
microbenchmark should land.

**What I changed.** Nothing in the derivation. The theoretical ceiling stays
derived from the clock the runtime reports, because that is the honest,
reproducible definition, and the report says plainly that the measured compute
ceiling exceeds it and why. This is precisely the case the NVML monitor exists
for: without the clock trace this would have looked like a counting bug, and I
would have gone looking for a factor of two that was never there.

## 2026-07-22 Shared memory tiling did not beat the naive GEMM

**Observation, not a failure.** Across every size measured, the shared memory
tiled GEMM is no faster than the naive one, and at the smaller sizes it is
slower:

| size | naive GFLOP/s | tiled GFLOP/s |
| --- | --- | --- |
| 512 | 1725 | 1499 |
| 1024 | 1986 | 1731 |
| 2048 | 2056 | 1734 |
| 4096 | 1584 | 1586 |

**Why this is credible rather than a bug.** Both kernels pass the same
correctness tests against cuBLAS, so they compute the right answer; the question
is only why the optimisation does not pay. The likely explanation is this card's
48 MB L2 cache. Tiling exists to cut DRAM traffic from redundant global loads,
but at 2048 the operands are 16 MB each and the whole working set sits in L2, so
the naive kernel's redundant loads are already served at cache speed and there is
no DRAM traffic left for tiling to remove. What tiling does add is a
`__syncthreads` per tile step and a shared memory round trip, which is pure cost
when the traffic it would have saved was never going to DRAM.

**Status.** This is a hypothesis with a mechanism, and the Nsight Compute pass is
what will confirm or kill it: if it is right, the naive kernel shows a high L2 hit
rate and DRAM traffic far below its theoretical byte count. Recorded here as
open, and it will be settled with counters rather than with the story that sounds
best. The register blocked rung, which raises reuse inside the register file
rather than through shared memory, does pay: 6028 GFLOP/s at 4096, near four
times the naive kernel.

## 2026-07-22 Small kernels measured cache bandwidth, not DRAM

**Symptom.** SAXPY at 4M elements reported 1583 GB/s of achieved bandwidth. The
card's theoretical DRAM bandwidth is 672 GB/s, so this is more than twice a
number that should be an upper bound.

**Root cause.** Same 48 MB L2. At 4M floats the two vectors are 32 MB together
and fit in L2 entirely, so after the warmup iterations the kernel is not touching
DRAM at all and the figure is L2 bandwidth wearing a DRAM label. The larger sizes
behave: at 16M and 64M elements, where the working set is 128 MB and 512 MB,
SAXPY settles to 560 to 566 GB/s, which is 93 percent of the measured copy
ceiling and entirely sensible.

**What it means for the report.** The "achieved bandwidth" column is computed
from the *theoretical* byte count, which assumes every byte comes from DRAM. When
it does not, the column overstates DRAM traffic. This is exactly the gap the spec
predicted between theoretical and measured byte counts, and it is why the
arithmetic intensity of every point is labelled with which byte count produced
it. The Nsight Compute DRAM counters give the real traffic and will move these
points to where they belong.

## 2026-07-22 The loader rejected the transpose, correctly and wrongly

**Symptom.** The analysis CLI refused the first real timing CSV:
`non positive values in 'achieved_gflops' at rows [20..35]`.

**Root cause.** Those sixteen rows are the transpose configurations, and a
transpose performs no floating point operations at all. Its achieved GFLOP/s is
exactly zero, which is a true measurement and the entire reason the kernel is in
the suite. My loader validated throughput as strictly positive, so it threw away
the one kernel whose defining property is doing no arithmetic.

**Fix.** Split the rule. Times stay strictly positive, because a kernel that took
zero time did not run. Throughput only has to be non negative and finite. Added a
regression test that feeds the loader a zero GFLOP transpose row and asserts it
survives, alongside one that still rejects a negative.

**Worth noting.** The loader was doing its job loudly, which is what it is for.
The rule it was enforcing was just wrong, and a silent loader would have left me
with a roofline quietly missing sixteen points.

## 2026-07-22 The GEMM tolerance was wrong, not the GEMM

**Symptom.** Every GEMM variant failed its correctness test at every size:
naive, tiled, register blocked, vectorized, and cuBLAS. Measured error against
a tolerance of 1e-4 was about 3.2e-3 at 512 and 2.3e-4 at 64.

**The clue that mattered.** All five variants reported the *identical* error to
sixteen digits, and one of the five was cuBLAS. Five independent implementations
do not share a bug. They do share a reference and an error metric, so the fault
had to be in one of those.

**Root cause.** My metric was a per element relative error, dividing each
deviation by `max(|expected|, 1e-3)`. With random operands in [-1, 1] each GEMM
output is a sum of k signed products, so a good fraction of the entries land near
zero through cancellation. Dividing a perfectly ordinary absolute error by one of
those near zero values produces a huge ratio that measures the cancellation, not
the kernel. The metric was reporting on the test data rather than on the code.

**Checking the arithmetic before changing anything.** Working backward from the
512 case: a reported 3.2e-3 against the 1e-3 floor means an absolute error near
3.2e-6, while a 512 term sum of products of values in [-1, 1] has magnitude
around 13. That is a true relative error near 2.5e-7, which is fp32 epsilon. The
kernels were right the whole time.

**Fix.** Switched to a normwise relative error: the largest absolute deviation
anywhere in the matrix over the largest magnitude in the reference. This is the
standard way to check a GEMM and it does not lose its meaning near zero. It still
catches genuine defects, because an indexing or tiling bug produces an error of
order 1 relative to the norm, not of order 1e-6.

**Why not just loosen the tolerance.** Because that would have hidden the real
problem behind a number chosen to make tests pass, and the tolerance would then
have been too loose to catch an actual bug at large k. The metric was wrong, so I
fixed the metric.

**Verification.** All 36 GPU correctness tests pass, covering GEMV and all five
GEMM rungs across square, rectangular, tile aligned, and deliberately unaligned
shapes, plus a 512 case checked against cuBLAS rather than a host triple loop.

## 2026-07-22 Checkpoint while the toolchain was still missing

Recorded at the pause before the toolchain was installed. Kept as written,
because the reasoning behind not writing kernels blind is the point of the entry,
and the next entries above show how it resolved.

Toolchain still not installed (re-probed: `nvcc`, `cl.exe`, `nsys`, `ncu`,
`perl`, and `ninja` all still absent), so Phases 1 through 4 remain blocked. What
is done and verified: the scaffold, the style gate, the Python analysis package
with 34 passing tests, the build system skeleton, CI, the docs set, and all three
report PDFs compiling. What is deliberately not done: the CUDA kernels. I am not
writing eight kernels I cannot compile, because the ground rule is correctness
before performance per kernel, and committing unverifiable GPU code would be the
opposite of that. They get written and compile tested one at a time once the
toolkit lands.
