environment:
    conda install -y pandas typer scikit-learn scikit-image colorcet matplotlib keras tensorflow plotly pywavelets seaborn datashader bokeh holoviews -c conda-forge umap-learn
	pip install -r requirements.txt
.PHONY: install

setup:
	python setup.py develop
.PHONY: install

test:
	pytest
.PHONY: install