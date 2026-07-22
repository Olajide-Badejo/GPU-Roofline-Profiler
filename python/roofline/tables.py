"""Booktabs LaTeX table fragments generated from result frames.

The report never hand types a number. These helpers turn a pandas frame into a
booktabs `tabular` fragment and write it into report/tables, where the main
document pulls it in with \\input. Writes are atomic (temp file then rename) so a
half written fragment can never be picked up by a concurrent latexmk run.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# Characters LaTeX treats specially, escaped so a stray underscore in a kernel
# name does not blow up the compile.
_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(text: str) -> str:
    """Escape LaTeX special characters in a cell string."""
    return "".join(_LATEX_ESCAPES.get(char, char) for char in text)


def dataframe_to_booktabs(
    frame: pd.DataFrame,
    column_format: str | None = None,
    float_format: str = "{:.2f}",
    headers: dict[str, str] | None = None,
) -> str:
    """Render a frame as a booktabs tabular fragment (no float, no caption).

    The caller wraps this in a table environment in the .tex if it wants a
    caption and label; keeping the fragment as a bare tabular makes it reusable
    inside subfigures and appendices. Column alignment defaults to left for the
    first column and right for the rest, which suits a label plus numbers layout.
    """
    headers = headers or {}
    n_cols = len(frame.columns)
    if column_format is None:
        column_format = "l" + "r" * (n_cols - 1)

    header_cells = [
        escape_latex(headers.get(str(col), str(col))) for col in frame.columns
    ]
    lines = [
        r"\begin{tabular}{" + column_format + "}",
        r"\toprule",
        " & ".join(header_cells) + r" \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        cells = [_format_cell(value, float_format) for value in row]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def write_table(
    frame: pd.DataFrame,
    out_path: str | Path,
    **kwargs: object,
) -> Path:
    """Render a frame to a booktabs fragment and write it atomically.

    Returns the path written. Extra keyword arguments pass through to
    :func:`dataframe_to_booktabs`.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = dataframe_to_booktabs(frame, **kwargs)  # type: ignore[arg-type]
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, out_path)
    return out_path


def _format_cell(value: object, float_format: str) -> str:
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        return float_format.format(value)
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    return escape_latex(str(value))
