PRX Intelligence manuscript (19 October 2026)

main.tex is the latest manuscript converted to the APS REVTeX 4.2 Physical
Review style recommended by PRX Intelligence. main-review.tex selects a
single-column referee version. supplementary.tex is the separate Supplemental
Material. Shared study-design.tex and compute-summary.tex must remain beside
these sources. Built PDFs are in build/. Shared figures remain in ../figures/.

From paper/prxintelligence/, build with Tectonic:
  tectonic --outdir build main.tex
  tectonic --outdir build supplementary.tex

Or build using a TeX distribution with REVTeX 4.2 installed:
  pdflatex -output-directory=build main.tex
  bibtex build/main
  pdflatex -output-directory=build main.tex
  pdflatex -output-directory=build main.tex
  pdflatex -output-directory=build supplementary.tex

For the optional single-column manuscript, use main-review.tex instead of
main.tex and main-review instead of main in the commands above.

The APS physrev style supplies numbered sections and bracketed citations;
longbibliography retains reference titles as required by PRX Intelligence.
REVTeX selects apsrev4-2 automatically. Do not add an explicit
bibliographystyle command, which can disable the title-display controls.
Figures/tables span both columns. Scientific content and results were
preserved during the format conversion.

Official guidelines:
https://journals.aps.org/prxintelligence/authors
https://journals.aps.org/revtex

Author review before submission:
- LLM-use disclosure corrected following author confirmation on 19 October 2026.
- The supplement now matches the main paper's author list and current studies.
- Compute figures are qualified estimates, recorded figures or planned allocations;
  no audited aggregate is claimed.
- The README and Data availability statement identify unreleased inputs.
