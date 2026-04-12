# Slides Scaffold

This directory contains the Beamer source for the OuroMaintain presentation.

## Build

From `slides/`:

```bash
latexmk -pdf main.tex
```

If `latexmk` is unavailable, compile with:

```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

## Structure

- `main.tex` is the deck entry point.
- `sections/` contains one file per slide group.
- `references.bib` holds the bibliography stub.

## Notes

- The deck is intentionally scaffolded with placeholder content.
- Replace placeholders with experiment outputs, charts, and screenshots once the training run completes.
