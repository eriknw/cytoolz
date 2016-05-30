PYTHON ?= python

inplace:
	$(PYTHON) setup.py build_ext --inplace --cython

test: inplace
	nosetests -s --with-doctest cytoolz/

clean:
	rm cytoolz/*.c cytoolz/*.so
