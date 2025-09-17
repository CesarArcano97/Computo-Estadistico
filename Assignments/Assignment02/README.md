# Assignment 2 - LaTeX Report

## Description
This directory contains all files related to Assignment 2 for the Statistical Computing course.

## Directory Structure
```
Assignment02/
├── src/                    # Source files
│   ├── main.tex           # Main LaTeX document
│   ├── sections/          # Individual sections
│   └── references.bib     # Bibliography file
├── figures/               # Figures and images
├── data/                  # Data files (if any)
├── output/               # Compiled PDFs and outputs
└── README.md            # This file
```

## Compilation Instructions

### Using VSCode
1. Open the `src/main.tex` file in VSCode
2. Ensure LaTeX Workshop extension is installed
3. Use Ctrl+Alt+B to build or click the LaTeX Workshop build button
4. The compiled PDF will appear in the `output/` directory

### Using Command Line
```bash
cd src/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Requirements
- LaTeX distribution (TeX Live recommended)
- Required packages (see main.tex preamble)
- VSCode with LaTeX Workshop extension (recommended)

## Submission
- Final PDF: `output/assignment2_report.pdf`
- Source code: All files in `src/` directory
- Due Date: [Add due date here]

---
**Author:** César Arcano  
**Course:** Statistical Computing - CIMAT Monterrey