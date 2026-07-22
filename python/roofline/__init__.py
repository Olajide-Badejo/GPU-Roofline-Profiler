"""Roofline analysis package.

Turns raw benchmark and profiler output from the RTX 5070 into the figures and
tables the LaTeX report consumes. The split mirrors the data flow:

* :mod:`roofline.model`    the roofline math itself (ceilings, ridge point,
                           arithmetic intensity, attainable performance).
* :mod:`roofline.loaders`  reading timing CSVs, NVML logs, and parsed Nsight
                           Compute output into validated pandas frames.
* :mod:`roofline.plotting` log-log roofline figures and the tensor core panel.
* :mod:`roofline.tables`   booktabs LaTeX fragments written into report/tables.

The math lives apart from the plotting on purpose: a wrong intensity or ceiling
formula silently invalidates every figure downstream, so that module is the one
with the heaviest unit test coverage.
"""

from __future__ import annotations

__version__ = "0.1.0"

from roofline import loaders, model, plotting, tables

__all__ = ["model", "loaders", "plotting", "tables", "__version__"]
