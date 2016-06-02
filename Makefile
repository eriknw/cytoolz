PYTHON ?= python

inplace:
	$(PYTHON) setup.py build_ext --inplace --cython

test: inplace
	nosetests -s --with-doctest cytoolz/

clean:
	rm -f cytoolz/*.c cytoolz/*.so cytoolz/*/*.c cytoolz/*/*.so
	rm -rf build/ cytoolz/__pycache__/ cytoolz/*/__pycache__/

curried:
	$(PYTHON) etc/generate_curried.py > cytoolz/curried/__init__.py
