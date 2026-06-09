# ICLR 2026 Submission Source

This directory contains the anonymous ICLR-style LaTeX source for the Certified TailGuard Best-of-N VLA paper.

Build from the repository root with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_iclr_submission.ps1
```

The script compiles `main.tex` with `latexmk -pdf` when available, falls back to `pdflatex`/`bibtex`, and copies the final PDF to:

```text
C:\Users\wangz\Downloads\certified-tailguard-bon-iclr-submission.pdf
```

Template provenance: `iclr2026_conference.sty`, `iclr2026_conference.bst`, and `math_commands.tex` were copied from the official ICLR 2026 template zip linked by the ICLR 2026 Author Guide.
