environment:
    conda install -y pandas typer scikit-learn scikit-image colorcet matplotlib keras tensorflow plotly pywavelets seaborn datashader bokeh holoviews
    pip install -r requirements.txt
.PHONY: install