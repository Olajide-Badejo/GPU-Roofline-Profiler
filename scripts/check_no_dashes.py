#!/usr/bin/env python3
"""Style gate: forbid em dashes and en dashes repo wide, and forbid the
LaTeX prose dash ligatures inside .tex sources.

The rule I hold myself to on this project (spec Section 4, rule 3):

  * U+2014 (em dash) and U+2013 (en dash) appear in no tracked file, ever,
    including plot labels, docstrings, and commit messages.
  * ASCII "--" and "---" appear in no .tex file, because TeX turns them into
    en and em dashes in the typeset output. Command line flags like
    "nvcc --version" are fine in shell scripts and docs, so the ASCII check
    is scoped to .tex only.

Exit code is non zero if any violation is found, so this doubles as a CI gate
and a local `make check-style` target. It scans git tracked files when run
inside a repository, and falls back to walking the tree otherwise.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Built from code points so this file does not trip its own check.
EM_DASH = chr(0x2014)
EN_DASH = chr(0x2013)

# The upstream build spec is an external input document, not an authored
# deliverable, so it is exempt from the authored prose rules.
SKIP_FILES = {"gpu_roofline_profiler_spec.md"}

# Files that are binary or generated and carry no authored prose. Extensions
# are lowercased before comparison.
BINARY_SUFFIXES = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".zip", ".gz", ".tar",
    ".exe", ".dll", ".so", ".dylib", ".o", ".obj", ".a", ".lib", ".bin",
    ".ncu-rep", ".nsys-rep", ".qdrep", ".sqlite", ".pyc",
}

SKIP_DIRS = {".git", "build", "__pycache__", ".venv", "venv", "node_modules"}


def tracked_files(root: Path) -> list[Path]:
    """Return git tracked files, or every file under root if git is absent."""
    try:
        out = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        files = [root / line for line in out.stdout.splitlines() if line]
        if files:
            return files
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    result: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and not any(part in SKIP_DIRS for part in path.parts):
            result.append(path)
    return result


def scan_file(path: Path) -> list[tuple[int, str]]:
    """Return (line_number, message) violations for one file."""
    if path.suffix.lower() in BINARY_SUFFIXES:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        # Not UTF-8 text; treat as binary and skip rather than guess.
        return []

    is_tex = path.suffix.lower() == ".tex"
    violations: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if EM_DASH in line:
            violations.append((lineno, "em dash (U+2014)"))
        if EN_DASH in line:
            violations.append((lineno, "en dash (U+2013)"))
        if is_tex and "--" in line:
            violations.append((lineno, 'ASCII "--" in .tex prose'))
    return violations


def main() -> int:
    """Scan tracked files and report dash violations; return process exit code."""
    root = Path(__file__).resolve().parent.parent
    total = 0
    for path in sorted(tracked_files(root)):
        if path.name in SKIP_FILES:
            continue
        for lineno, message in scan_file(path):
            rel = path.relative_to(root)
            print(f"{rel}:{lineno}: {message}")
            total += 1
    if total:
        print(f"\ncheck_no_dashes: {total} violation(s) found", file=sys.stderr)
        return 1
    print("check_no_dashes: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
