#!/usr/bin/env python
"""
Cython-compatible wrapper for doctest.testmod().

Usage example, assuming a Cython module mymod.pyx is compiled.
This is run from the command line, passing a command to Python:
python -c "import cydoctest, mymod; cydoctest.module_doctest(mymod)"

(This still won't let a Cython module run its own doctests
when called with "python mymod.py", but it's pretty close.
Further options can be passed to module_doctest() as desired, e.g.
verbose=True.)
"""

# The utilities in this file were obtained from:
# https://github.com/cython/cython/wiki/FAQ
# #how-can-i-run-doctests-in-cython-code-pyx-files

import doctest
import inspect


def _from_module(module, object):
    """
    Return true if the given object is defined in the given module.
    """
    if module is None:
        return True
    elif inspect.getmodule(object) is not None:
        return module is inspect.getmodule(object)
    elif inspect.isfunction(object):
        return module.__dict__ is object.func_globals
    elif inspect.isclass(object):
        return module.__name__ == object.__module__
    elif hasattr(object, '__module__'):
        return module.__name__ == object.__module__
    elif isinstance(object, property):
        return True  # [XX] no way not be sure.
    else:
        raise ValueError("object must be a class or function")


def fix_module_doctest(module):
    """
    Extract docstrings from cython functions, that would be skipped by doctest
    otherwise.
    """
    module.__test__ = {}
    for name in dir(module):
        value = getattr(module, name)
        if (inspect.isbuiltin(value) and isinstance(value.__doc__, str) and
                _from_module(module, value)):
            module.__test__[name] = value.__doc__


def module_doctest(m, *args, **kwargs):
    """
    Fix a Cython module's doctests, then call doctest.module_doctest()

    All other arguments are passed directly to doctest.module_doctest().

    Return True on success, False on failure.
    """
    fix_module_doctest(m)
    return doctest.testmod(m, *args, **kwargs).failed == 0


if __name__ == '__main__':
    import cytoolz
    import cytoolz.dicttoolz
    import cytoolz.functoolz
    import cytoolz.itertoolz
    import cytoolz.recipes

    passed = True
    passed &= module_doctest(cytoolz.dicttoolz)
    passed &= module_doctest(cytoolz.functoolz)
    passed &= module_doctest(cytoolz.itertoolz)
    passed &= module_doctest(cytoolz.recipes)
    if not passed:
        exit(1)
