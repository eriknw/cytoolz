PYTHON ?= python

inplace:
	$(PYTHON) setup.py build_ext --inplace

test: inplace
	nosetests -s --with-doctest cytoolz/
