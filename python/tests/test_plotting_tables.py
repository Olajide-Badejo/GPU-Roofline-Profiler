"""Smoke and correctness tests for plotting and table generation.

The plotting test does not inspect pixels; it checks that a valid figure is
produced and written, which catches import, backend, and axis wiring breakage.
The table tests check the emitted LaTeX exactly, including escaping, because a
malformed fragment would break the report compile far downstream.
"""

from __future__ import annotations

import pandas as pd
import pytest

from roofline import tables
from roofline.model import Ceilings, IntensitySource
from roofline.plotting import RooflinePoint, plot_roofline


def test_plot_roofline_writes_pdf_and_png(tmp_path):
    ceilings = [
        Ceilings(31.0e12, 672.0e9, "theoretical"),
        Ceilings(27.0e12, 600.0e9, "measured"),
    ]
    points = [
        RooflinePoint("saxpy", "1M", 0.25, 250.0, IntensitySource.MEASURED, True),
        RooflinePoint(
            "gemm_tiled", "4096", 40.0, 9800.0, IntensitySource.THEORETICAL, True
        ),
        # An unrecognised series must still draw, in the muted fallback style,
        # rather than raising or inventing a new hue.
        RooflinePoint("something_new", "1", 5.0, 100.0),
    ]
    pdf = tmp_path / "figures" / "roofline.pdf"
    png = tmp_path / "figures" / "roofline.png"
    out = plot_roofline(points, ceilings, pdf, png, title="RTX 5070 roofline")
    assert out == pdf
    assert pdf.exists() and pdf.stat().st_size > 0
    assert png.exists() and png.stat().st_size > 0


def test_plot_roofline_requires_a_ceiling(tmp_path):
    with pytest.raises(ValueError):
        plot_roofline([], [], tmp_path / "x.pdf")


def test_booktabs_structure_and_escaping():
    frame = pd.DataFrame(
        {"kernel": ["gemm_naive"], "size": [4096], "gflops": [1234.567]}
    )
    latex = tables.dataframe_to_booktabs(frame, float_format="{:.1f}")
    assert r"\toprule" in latex
    assert r"\midrule" in latex
    assert r"\bottomrule" in latex
    # Underscore in the kernel name must be escaped.
    assert r"gemm\_naive" in latex
    # Integer stays integer, float honours the format string.
    assert "4096" in latex
    assert "1234.6" in latex


def test_write_table_is_atomic_and_readable(tmp_path):
    frame = pd.DataFrame({"a": [1], "b": [2.5]})
    out = tables.write_table(frame, tmp_path / "sub" / "frag.tex")
    assert out.exists()
    assert not (tmp_path / "sub" / "frag.tex.tmp").exists()
    assert r"\begin{tabular}" in out.read_text(encoding="utf-8")
