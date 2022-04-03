environment:
	conda install -y tensorflow==2.6.0
	pip install -r requirements.txt --user
	pip install datashader==0.13.0
	pip install "holoviews[recommended]" --user
.PHONY: install
