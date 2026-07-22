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
