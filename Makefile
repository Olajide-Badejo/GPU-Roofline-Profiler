# Convenience targets. On Windows the PowerShell scripts under scripts/ are the
# primary entry points; this Makefile mirrors them for a POSIX shell (WSL2 or
# msys2) and for CI. The CUDA build targets need the toolkit; the style and
# Python targets do not.

PYTHON ?= python
BUILD_DIR ?= build

.PHONY: help check-style dash-lint ruff test-python configure build reports \
        clean

help:
	@echo "targets:"
	@echo "  check-style   dash lint plus ruff over the Python package"
	@echo "  test-python   run the pytest suite"
	@echo "  configure     cmake configure (needs the CUDA toolkit)"
	@echo "  build         cmake build (needs the CUDA toolkit)"
	@echo "  reports       build all report PDFs"
	@echo "  clean         remove the build directory"

check-style: dash-lint ruff

dash-lint:
	$(PYTHON) scripts/check_no_dashes.py

ruff:
	$(PYTHON) -m ruff check python

test-python:
	$(PYTHON) -m pytest python -q

configure:
	cmake -S . -B $(BUILD_DIR) -G Ninja

build: configure
	cmake --build $(BUILD_DIR)

reports:
	$(MAKE) -C report pdf
	$(MAKE) -C report_debug pdf

clean:
	rm -rf $(BUILD_DIR)
