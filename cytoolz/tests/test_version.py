""" Version number is defined twice: in "cytoolz/__init__.py" and "setup.py".

This test verifies they are the same and reports a helpful message if not.
"""
# A sane person would have a separate "VERSION" file or what have you.  Oh well.
import cytoolz
import imp
import os.path


def test_versions_consistent():
    # This assumes `nosetests` is run from the same diretory as "setup.py"
    assert os.path.exists('setup.py'), (
        '"setup.py" not found.  Are you sure you are running `nosetests` '
        '(or equivalent) from the root directory of the github repo?')

    setup = imp.load_source('cytoolz_setup', 'setup.py')
    assert cytoolz.__version__ == setup.VERSION, (
        'Version in cytoolz (%s) does not match version in setup.py (%s)'
        % (cytoolz.__version__, setup.VERSION))
