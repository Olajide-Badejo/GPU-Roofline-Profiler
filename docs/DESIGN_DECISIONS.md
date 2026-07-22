# Design decisions

Every independent judgment call this document did not resolve for me, recorded
with its reasoning so a reader can see why the project looks the way it does.

## DD-1. Build in place in the existing directory

The spec assumes a fresh empty repo. I initialized git in the directory that
already held the spec file rather than nesting a `gpu-roofline-profiler/`
subfolder, so paths in the spec map one to one onto the repo root. The spec
document itself stays at the root as `gpu_roofline_profiler_spec.md` for
reference.

## DD-2. Environment route: native Windows (pending toolchain install)

The spec makes native Windows the default and WSL2 the fallback. I am holding to
native Windows because the driver and Nsight tooling are first party there and
there is no virtualization layer between profiler and GPU. This is only a real
decision once the toolchain is installed; see the open item below.

## DD-3. CUDA Toolkit version target: 13.3 (resolved)

`nvidia-smi` reported CUDA UMD 13.3 and driver 610.62. sm_120 (Blackwell GB205)
needs CUDA 12.8 at minimum; the spec prefers 12.9 or newer. Because the driver
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

## Resolved: developer toolchain install

The machine had the GPU and driver but no CUDA Toolkit, no MSVC, and no Nsight
tools. Per spec Section 0 I stopped and asked rather than running elevated
installers unattended. The owner installed them. Verified present: CUDA 13.3.73,
MSVC 14.44.35207 (Build Tools 2022), Nsight Compute 2026.2.1, Nsight Systems
2026.1.3, Ninja 1.12.0, Strawberry Perl 5.42.2.

Because a shell inherits its environment at startup, tool sessions opened before
the install still carry a stale PATH, and `cl.exe` only exists inside a Visual
Studio developer environment. `scripts/dev_env.ps1` handles all of it: re-reads
PATH from the registry, imports `vcvars64`, and adds Nsight Systems, which its
installer leaves off PATH.
