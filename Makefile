environment:
	conda install -y tensorflow==2.6.0
	pip install -r requirements.txt
	pip install datashader==0.13.0
	pip install "holoviews[recommended]"
    pip install Jinja2==3.0.*
.PHONY: install
