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

## DD-3. CUDA Toolkit version target

`nvidia-smi` reports CUDA UMD 13.3 and driver 610.62. sm_120 (Blackwell GB205)
needs CUDA 12.8 at minimum; the spec prefers 12.9 or newer. Because the driver
already speaks CUDA 13.3, I plan to target a CUDA 13.x toolkit rather than 12.9,
so the toolkit and driver match and I get the newest Blackwell code generation.
`CMAKE_CUDA_ARCHITECTURES` stays auto detected with `120` as the explicit
fallback regardless of toolkit version.

## DD-4 (open). latexmk vs direct pdflatex

`latexmk` is broken on this machine because MiKTeX has no Perl engine. Two ways
forward: install Strawberry Perl so `latexmk` works as the spec assumes, or
drive the report build with a direct `pdflatex` + `bibtex` + `pdflatex` sequence
in the report Makefiles and drop the `latexmk` dependency. I lean toward
installing Perl so the report Makefiles stay conventional, but this is bundled
into the toolchain install decision below and will be resolved there.

## Open decision: developer toolchain install

Blocking all compilation. The machine has the GPU and driver but no CUDA
Toolkit, no MSVC, and no Nsight tools. Installing them needs elevated
permissions, so per spec Section 0 this is a fork I bring to the owner rather
than resolve alone. Resolution and the exact installed versions will be recorded
here and in the engineering log once chosen.
