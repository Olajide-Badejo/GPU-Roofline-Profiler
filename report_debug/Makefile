# Build the debug report. Same latexmk-with-fallback approach as the main
# report: latexmk when a Perl engine is present, a direct pdflatex plus bibtex
# sequence otherwise.

MAIN := debug_report
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
	$(PDFLATEX) $(MAIN).tex

clean:
	$(LATEXMK) -C $(MAIN).tex || rm -f $(MAIN).aux $(MAIN).log $(MAIN).out \
		$(MAIN).toc $(MAIN).fls $(MAIN).fdb_latexmk
