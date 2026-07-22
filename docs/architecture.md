# Architecture

The data flow, from a kernel launch to a line in the PDF.

## Overview

```
configs/sweep.yaml
        |
        v
  C++ driver (src/main.cpp)  ----launches---->  kernels (src/kernels/*.cu)
        |         |                                   ^
        |         | NVML monitor thread               | cudaEvent timing
        |         v                                    |
        |   nvml_<run>.csv                             |
        v                                              |
  results/raw/<utc>_<sha>/timing.csv, manifest.json <--
        |
        |   (Nsight passes wrap the same driver)
        |     nsys -> timeline reports
        |     ncu  -> per kernel counters -> ncu_<...>.csv
        v
  python/cli.py  ->  roofline.loaders  ->  roofline.model
        |                                        |
        |                                        v
        |                              roofline.plotting -> report/figures/*.pdf,*.png
        |                              roofline.tables   -> report/tables/*.tex
        v
  report/main.tex  --(latexmk or pdflatex)-->  report/main.pdf
```

## Components

- **Sweep config** (`configs/sweep.yaml`): the single source of what runs. The
  driver resolves it into a list of (kernel, config) cells and copies the
  resolved config into each run manifest.
- **Driver** (`src/main.cpp`): allocates, checks free VRAM per cell, launches
  kernels through the timing harness, runs the correctness check before timing,
  writes `timing.csv` atomically, and drives the progress bar. An NVML monitor
  thread samples power and clocks for the whole run.
- **Kernels** (`src/kernels/`): one `.cu` per family, tile sizes as template
  parameters. Each has a CPU or cuBLAS reference for its correctness test.
- **Profiling** (`src/profiling/`): NVML monitor, device peak derivation and
  measurement, and NVTX helpers for the Nsight Systems timeline.
- **Analysis** (`python/`): loaders validate the CSVs, the model computes
  ceilings and intensities, plotting draws the rooflines, tables writes booktabs
  fragments. `cli.py` runs all of it with one command.
- **Reports** (`report/`, `report_debug/`): LaTeX that pulls in the generated
  figures and tables. Nothing in them is a hand typed number.

## Reproducibility

Every run writes `results/raw/<utc timestamp>_<git sha>/` with a `manifest.json`
capturing the resolved config, driver and CUDA versions, an `nvidia-smi`
snapshot, and the Nsight tool versions. The driver and profiling scripts skip
cells whose output already carries a `.done` marker, so an interrupted pass
resumes rather than restarting. `--force` redoes a cell. The report's environment
appendix is generated from these manifests at build time.
