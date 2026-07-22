# Build the main report. The spec's intended tool is latexmk; on this machine
# MiKTeX's latexmk needs a Perl engine that was not installed at first, so the
# default target tries latexmk and falls back to a direct pdflatex plus bibtex
# sequence that needs no Perl. Once Strawberry Perl is present, latexmk drives
# the build and the fallback is never reached.

MAIN := main
LATEXMK := latexmk
PDFLATEX := pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX := bibtex

.PHONY: pdf latexmk manual clean

pdf:
	$(LATEXMK) -pdf -interaction=nonstopmode $(MAIN).tex || $(MAKE) manual

latexmk:
	$(LATEXMK) -pdf -interaction=nonstopmode $(MAIN).tex

manual:
	$(PDFLATEX) $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(PDFLATEX) $(MAIN).tex
	$(PDFLATEX) $(MAIN).tex

clean:
	$(LATEXMK) -C $(MAIN).tex || rm -f $(MAIN).aux $(MAIN).log $(MAIN).out \
		$(MAIN).toc $(MAIN).bbl $(MAIN).blg $(MAIN).fls $(MAIN).fdb_latexmk
