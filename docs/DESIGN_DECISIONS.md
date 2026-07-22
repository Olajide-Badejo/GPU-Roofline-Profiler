# Design decisions

Every judgment call worth defending, recorded with its reasoning so a reader can
see why the project looks the way it does rather than having to guess.

## DD-1. Build in place in the existing directory

I initialized git in the project directory itself rather than nesting a
`gpu-roofline-profiler/` subfolder inside it, so the documented layout maps one
to one onto the repository root.

## DD-2. Environment route: native Windows (pending toolchain install)

I made native Windows the default and WSL2 the fallback. I am holding to
native Windows because the driver and Nsight tooling are first party there and
there is no virtualization layer between profiler and GPU. This is only a real
decision once the toolchain is installed; see the open item below.

## DD-3. CUDA Toolkit version target: 13.3 (resolved)

`nvidia-smi` reported CUDA UMD 13.3 and driver 610.62. sm_120 (Blackwell GB205)
needs CUDA 12.8 at minimum, and I wanted 12.9 or newer. Because the driver
already spoke CUDA 13.3, I targeted a CUDA 13.x toolkit rather than 12.9, so the
toolkit and driver match and I get the newest Blackwell code generation.
Installed and verified: **CUDA 13.3.73**.

The cost of that choice showed up immediately: CUDA 13 removed `clockRate` and
`memoryClockRate` from `cudaDeviceProp`, so the peak derivation reads them
through `cudaDeviceGetAttribute` instead. That is recorded in the engineering log
with the verification that the resulting peaks match the vendor reference. I
consider the trade worth it: matching the driver avoids a whole class of version
skew problems, and the attribute query is the supported API going forward.

## DD-4. latexmk restored by installing Perl (resolved)

Strawberry Perl 5.42.2 is installed, so MiKTeX's `latexmk` works and the report
Makefiles can stay conventional. I kept the direct `pdflatex` plus `bibtex`
fallback in each Makefile and in `scripts/build_reports.ps1` anyway, because it
costs nothing and it means the reports still build on a machine without Perl.

## DD-5. CUDA architecture selected before `project()`

`CMAKE_CUDA_ARCHITECTURES` is set before the `project()` call, not after. Calling
`project()` with the CUDA language enabled installs a default value, so the usual
`if(NOT DEFINED ...)` guard placed afterward never fires and the build silently
targets the wrong architecture. Preference order is `CUDAARCHS` from the
environment, then `native`, then `120` for a GPU free CI runner.

## DD-6. yaml-cpp compatibility shim for CMake 4

yaml-cpp 0.8.0 declares a `cmake_minimum_required` below 3.5, which CMake 4.3.1
rejects. I set `CMAKE_POLICY_VERSION_MINIMUM` to 3.5 around the
`FetchContent_MakeAvailable` call and unset it immediately after, so the shim
applies only to the fetched dependency. The alternatives were vendoring a patched
copy (a maintenance burden) or hand writing a YAML subset parser (error prone,
and a lot of work to dodge one upstream version declaration).

## DD-7. Adaptive batch sizing in the timing harness

The harness measures one launch, then chooses how many launches to put between a
pair of CUDA events so a batch lasts roughly 50 ms, capped by the configured
maximum. A single fixed count cannot serve this suite: SAXPY at a million
elements takes tens of microseconds, so timing one launch at a time would measure
mostly launch overhead and event resolution, while a naive GEMM at 4096 takes
tens of milliseconds, so fifty launches per batch would mean minutes per cell.
With the calibration in place the full 62 cell sweep runs in 31 seconds.

## DD-8. Roofline colour, and what the current scheme gives up

A roofline is a scatter, so any two series can end up adjacent on screen and the
palette has to hold up for every pair, not just neighbouring ones.

An earlier version drew the GEMM ladder as what it structurally is, an ordered
progression from naive to cuBLAS, using one blue hue from light to dark. That had
a real advantage: darker read as more optimised without consulting the legend.

The current scheme uses distinct categorical hues per rung (red, yellow, pink,
brown) because I preferred them visually. The trade is explicit: the ladder's
ordering is no longer carried by shade and now rests entirely on the legend order
and the direct labels. To keep that from costing accessibility, every series also
carries a distinct marker shape, so identity is never carried by colour alone and
the figure still survives colourblind readers and greyscale printing. The
measured ceiling is lemon, which is pale against a near white surface, so that
line is drawn thicker and given a dark outline rather than left to fend for
itself.

One honest gap: the palette validator I would normally run ships as a Node script
and this machine has no Node, so I could not execute it. Rather than eyeball
colour separation, which is exactly what such tooling exists to prevent, I kept
to documented values and treated the marker shapes as required secondary encoding
rather than decoration. If Node is installed later, the next step is to run the
validator over the exact set in use and re-step anything that fails.

## Resolved: developer toolchain install

The machine had the GPU and driver but no CUDA Toolkit, no MSVC, and no Nsight
tools. I installed them deliberately rather than letting a script run elevated
installers unattended. Verified present: CUDA 13.3.73,
MSVC 14.44.35207 (Build Tools 2022), Nsight Compute 2026.2.1, Nsight Systems
2026.1.3, Ninja 1.12.0, Strawberry Perl 5.42.2.

Because a shell inherits its environment at startup, tool sessions opened before
the install still carry a stale PATH, and `cl.exe` only exists inside a Visual
Studio developer environment. `scripts/dev_env.ps1` handles all of it: re-reads
PATH from the registry, imports `vcvars64`, and adds Nsight Systems, which its
installer leaves off PATH.
