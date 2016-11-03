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

copytests:
	for f in ../toolz/toolz/tests/test*py; \
	do \
		if [[ $$f == *test_utils* ]]; then continue ; fi;  \
		newf=`echo $$f | sed 's/...toolz.toolz/cytoolz/g'`; \
		sed -e 's/toolz/cytoolz/g' -e 's/itercytoolz/itertoolz/' \
			-e 's/dictcytoolz/dicttoolz/g' -e 's/funccytoolz/functoolz/g' \
			$$f > $$newf; \
		echo $$f $$newf; \
	done
