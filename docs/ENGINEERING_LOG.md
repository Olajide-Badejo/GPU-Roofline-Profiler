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

## 2026-07-22 Status at the end of the first session

Toolchain still not installed (re-probed: `nvcc`, `cl.exe`, `nsys`, `ncu`,
`perl`, and `ninja` all still absent), so Phases 1 through 4 remain blocked. What
is done and verified: the scaffold, the style gate, the Python analysis package
with 34 passing tests, the build system skeleton, CI, the docs set, and all three
report PDFs compiling. What is deliberately not done: the CUDA kernels. I am not
writing eight kernels I cannot compile, because the ground rule is correctness
before performance per kernel, and committing unverifiable GPU code would be the
opposite of that. They get written and compile tested one at a time once the
toolkit lands.
