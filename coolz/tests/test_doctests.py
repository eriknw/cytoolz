from coolz.utils import testmod

import coolz
import coolz.dicttoolz
import coolz.functoolz
import coolz.itertoolz
import coolz.dicttoolz.core
import coolz.functoolz.core
import coolz.itertoolz.core


# This currently doesn't work.  Use `cydoctest.py` instead.
def test_doctest():
    testmod(coolz)
    testmod(coolz.dicttoolz)
    testmod(coolz.functoolz)
    testmod(coolz.itertoolz)
    testmod(coolz.dicttoolz.core)
    testmod(coolz.functoolz.core)
    testmod(coolz.itertoolz.core)
