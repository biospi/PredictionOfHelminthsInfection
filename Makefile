environment:
	conda install -y pandas==1.4.1 typer==0.3.2 scikit-learn==0.23.2 scikit-image==0.16.2 colorcet==2.0.6
	matplotlib==3.3.3 keras==2.6.0 tensorflow==2.6.0 plotly==4.2.1 umap-learn==0.51
	pywavelets==1.1.1 seaborn==0.11.1 datashader==0.13.0 bokeh==2.3.3 holoviews==1.14.5
	pip install -r requirements.txt
	python setup.py develop
.PHONY: install

setup:
	python setup.py develop
.PHONY: install

test:
	pytest
.PHONY: install