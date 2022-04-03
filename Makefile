environment:
	conda install -y keras tensorflow
	pip install -r requirements.txt
	pip install datashader==0.13.0
	pip install "holoviews[recommended]"
	python setup.py develop
.PHONY: install
