#cython: embedsignature=True
from cpython.dict cimport PyDict_Check
from cytoolz.dicttoolz cimport c_merge_with

__all__ = ['merge_with']


def merge_with(func, *dicts):
    """
    Merge dictionaries and apply function to combined values

    A key may occur in more than one dict, and all values mapped from the key
    will be passed to the function as a list, such as func([val1, val2, ...]).

    >>> merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20})
    {1: 11, 2: 22}

    >>> merge_with(first, {1: 1, 2: 2}, {2: 20, 3: 30})  # doctest: +SKIP
    {1: 1, 2: 2, 3: 30}

    See Also:
        merge
    """
    if len(dicts) == 0:
        raise TypeError
    if len(dicts) == 1 and not PyDict_Check(dicts[0]):
        dicts = dicts[0]

    return c_merge_with(func, dicts)
