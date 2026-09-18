SHELL= /bin/bash
PYTHON ?= python
TOOLZ = ../toolz/toolz

# Rewrite toolz source as cytoolz source
TOOLZ2CYTOOLZ = sed -e 's/toolz/cytoolz/g' -e 's/itercytoolz/itertoolz/g' \
	-e 's/dictcytoolz/dicttoolz/g' -e 's/funccytoolz/functoolz/g'

# Tests copied from toolz, except those cytoolz maintains by hand
NOCOPY = test_utils.py test_curried_doctests.py test_tlz.py
TESTS = $(filter-out $(NOCOPY),$(notdir $(wildcard $(TOOLZ)/tests/test*.py)))

# cytoolz-specific edits to the copied tests; `copytests` re-applies it
TESTPATCH = copytests.patch

inplace:
	$(PYTHON) setup.py build_ext --inplace --cython

test: inplace
	pytest -s --doctest-modules cytoolz/
	echo 'cimport cytoolz ; from cytoolz.functoolz cimport memoize' > try_cimport_cytoolz.pyx
	echo 'import setuptools, Cython.Build ; setuptools.setup(ext_modules=Cython.Build.cythonize("try_cimport_cytoolz.pyx"))' > try_cimport_cytoolz_setup.py
	python try_cimport_cytoolz_setup.py build_ext --inplace
	python -c 'import try_cimport_cytoolz'
	rm try_cimport_cytoolz*
	python -c 'import cytoolz ; print(f"{cytoolz.__version__=}")'

clean:
	rm -f cytoolz/*.c cytoolz/*.so cytoolz/*/*.c cytoolz/*/*.so
	rm -rf build/ __pycache__/ cytoolz/__pycache__/ cytoolz/*/__pycache__/

curried:
	$(TOOLZ2CYTOOLZ) $(TOOLZ)/curried/__init__.py > cytoolz/curried/__init__.py

# Copy tests from toolz, then re-apply $(TESTPATCH)
copytests:
	for f in $(TESTS); do $(TOOLZ2CYTOOLZ) $(TOOLZ)/tests/$$f > cytoolz/tests/$$f; done
	git apply --verbose --allow-empty $(TESTPATCH)

# Save cytoolz-specific edits to the copied tests as $(TESTPATCH)
testpatch:
	: > $(TESTPATCH)
	for f in $(TESTS); do \
		$(TOOLZ2CYTOOLZ) $(TOOLZ)/tests/$$f \
		| diff -u --label a/cytoolz/tests/$$f --label b/cytoolz/tests/$$f - cytoolz/tests/$$f \
		>> $(TESTPATCH) || true; \
	done
	git apply --stat --allow-empty $(TESTPATCH)
