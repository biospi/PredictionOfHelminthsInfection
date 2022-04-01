environment:
	conda install -y pandas typer scikit-learn scikit-image colorcet keras tensorflow plotly pywavelets seaborn datashader bokeh holoviews
	pip install -r requirements.txt
	python setup.py develop
.PHONY: install

setup:
	python setup.py develop
.PHONY: install

test:
	pytest
.PHONY: install