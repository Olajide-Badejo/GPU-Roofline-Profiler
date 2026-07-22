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
