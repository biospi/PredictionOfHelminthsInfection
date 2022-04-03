environment:
	conda install -y keras tensorflow
	pip install -r requirements.txt --user
	pip install datashader==0.13.0 --user
	pip install "holoviews[recommended]" --user
	python setup.py develop
.PHONY: install
