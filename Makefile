environment:
	pip install -r requirements.txt
	python setup.py develop
.PHONY: install

setup:
	python setup.py develop
.PHONY: install

test:
	pytest
.PHONY: install