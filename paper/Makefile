# Build all standalone figures first, then the manuscript.
FIGSRC := $(wildcard figures/fig_*.tex)
FIGPDF := $(FIGSRC:.tex=.pdf)

.PHONY: all figures clean

all: main.pdf concepts.pdf

figures: $(FIGPDF)

figures/%.pdf: figures/%.tex figures/tikz-common.tex
	latexmk -pdf -cd -interaction=nonstopmode -quiet $<

main.pdf: main.tex refs.bib $(FIGPDF)
	latexmk -pdf -interaction=nonstopmode -quiet main.tex

concepts.pdf: concepts.tex $(FIGPDF)
	latexmk -pdf -interaction=nonstopmode -quiet concepts.tex

clean:
	latexmk -C main.tex
	latexmk -C concepts.tex
	rm -f figures/*.aux figures/*.log figures/*.fls figures/*.fdb_latexmk figures/*.synctex.gz
