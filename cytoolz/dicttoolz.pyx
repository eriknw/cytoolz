from cpython.dict cimport (PyDict_Check, PyDict_Copy, PyDict_GetItem,
                           PyDict_Merge, PyDict_New, PyDict_SetItem,
                           PyDict_Update)
from cpython.exc cimport PyErr_Clear, PyErr_GivenExceptionMatches, PyErr_Occurred
from cpython.list cimport PyList_Append, PyList_New
from cpython.ref cimport PyObject

# Locally defined bindings that differ from `cython.cpython` bindings
from .cpython cimport PyObject_GetItem


__all__ = ['merge', 'merge_with', 'valmap', 'keymap', 'valfilter', 'keyfilter',
           'assoc', 'get_in']  # 'update_in']


cdef dict c_merge(object dicts):
    cdef dict rv
    rv = PyDict_New()
    for d in dicts:
        PyDict_Update(rv, d)
    return rv


def merge(*dicts):
    """ Merge a collection of dictionaries

    >>> merge({1: 'one'}, {2: 'two'})
    {1: 'one', 2: 'two'}

    Later dictionaries have precedence

    >>> merge({1: 2, 3: 4}, {3: 3, 4: 4})
    {1: 2, 3: 3, 4: 4}

    See Also:
        merge_with
    """
    if len(dicts) == 1 and not PyDict_Check(dicts[0]):
        dicts = dicts[0]
    return c_merge(dicts)


cdef dict c_merge_with(object func, object dicts):
    cdef dict result, rv
    cdef list seq
    cdef object k, v
    cdef PyObject *obj
    result = PyDict_New()
    rv = PyDict_New()
    for d in dicts:
        for k, v in d.iteritems():
            obj = PyDict_GetItem(result, k)
            if obj is NULL:
                seq = PyList_New(0)
                PyList_Append(seq, v)
                PyDict_SetItem(result, k, seq)
            else:
                PyList_Append(<object>obj, v)
    for k, v in result.iteritems():
        PyDict_SetItem(rv, k, func(v))
    return rv


def merge_with(func, *dicts):
    """ Merge dictionaries and apply function to combined values

    A key may occur in more than one dict, and all values mapped from the key
    will be passed to the function as a list, such as func([val1, val2, ...]).

    >>> merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20})
    {1: 11, 2: 22}

    >>> merge_with(first, {1: 1, 2: 2}, {2: 20, 3: 30})  # doctest: +SKIP
    {1: 1, 2: 2, 3: 30}

    See Also:
        merge
    """
    if len(dicts) == 1 and not PyDict_Check(dicts[0]):
        dicts = dicts[0]
    return c_merge_with(func, dicts)


cpdef dict valmap(object func, object d):
    """ Apply function to values of dictionary

    >>> bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    >>> valmap(sum, bills)  # doctest: +SKIP
    {'Alice': 65, 'Bob': 45}

    See Also:
        keymap
    """
    cdef dict rv
    rv = PyDict_New()
    for k, v in d.iteritems():
       PyDict_SetItem(rv, k, func(v))
    return rv


cpdef dict keymap(object func, object d):
    """ Apply function to keys of dictionary

    >>> bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    >>> keymap(str.lower, bills)  # doctest: +SKIP
    {'alice': [20, 15, 30], 'bob': [10, 35]}

    See Also:
        valmap
    """
    cdef dict rv
    rv = PyDict_New()
    for k, v in d.iteritems():
       PyDict_SetItem(rv, func(k), v)
    return rv


cpdef dict valfilter(object predicate, object d):
    """ Filter items in dictionary by value

    >>> iseven = lambda x: x % 2 == 0
    >>> d = {1: 2, 2: 3, 3: 4, 4: 5}
    >>> valfilter(iseven, d)
    {1: 2, 3: 4}

    See Also:
        keyfilter
        valmap
    """
    cdef dict rv
    rv = PyDict_New()
    for k, v in d.iteritems():
        if predicate(v):
            PyDict_SetItem(rv, k, v)
    return rv


cpdef dict keyfilter(object predicate, object d):
    """ Filter items in dictionary by key

    >>> iseven = lambda x: x % 2 == 0
    >>> d = {1: 2, 2: 3, 3: 4, 4: 5}
    >>> keyfilter(iseven, d)
    {2: 3, 4: 5}

    See Also:
        valfilter
        keymap
    """
    cdef dict rv
    rv = PyDict_New()
    for k, v in d.iteritems():
        if predicate(k):
            PyDict_SetItem(rv, k, v)
    return rv


cpdef dict assoc(object d, object key, object value):
    """
    Return a new dict with new key value pair

    New dict has d[key] set to value. Does not modify the initial dictionary.

    >>> assoc({'x': 1}, 'x', 2)
    {'x': 2}
    >>> assoc({'x': 1}, 'y', 3)   # doctest: +SKIP
    {'x': 1, 'y': 3}
    """
    cdef dict rv
    rv = PyDict_Copy(d)
    PyDict_SetItem(rv, key, value)
    return rv


cdef tuple _get_in_exceptions = (KeyError, IndexError, TypeError)


cpdef object get_in(object keys, object coll, object default=None, bint no_default=False):
    """
    Returns coll[i0][i1]...[iX] where [i0, i1, ..., iX]==keys.

    If coll[i0][i1]...[iX] cannot be found, returns ``default``, unless
    ``no_default`` is specified, then it raises KeyError or IndexError.

    ``get_in`` is a generalization of ``operator.getitem`` for nested data
    structures such as dictionaries and lists.

    >>> transaction = {'name': 'Alice',
    ...                'purchase': {'items': ['Apple', 'Orange'],
    ...                             'costs': [0.50, 1.25]},
    ...                'credit card': '5555-1234-1234-1234'}
    >>> get_in(['purchase', 'items', 0], transaction)
    'Apple'
    >>> get_in(['name'], transaction)
    'Alice'
    >>> get_in(['purchase', 'total'], transaction)
    >>> get_in(['purchase', 'items', 'apple'], transaction)
    >>> get_in(['purchase', 'items', 10], transaction)
    >>> get_in(['purchase', 'total'], transaction, 0)
    0
    >>> get_in(['y'], {}, no_default=True)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    KeyError: 'y'

    See Also:
        itertoolz.get
        operator.getitem
    """
    cdef object item
    cdef PyObject *obj
    for item in keys:
        obj = PyObject_GetItem(coll, item)
        if obj is NULL:
            item = <object>PyErr_Occurred()
            if no_default or not PyErr_GivenExceptionMatches(item, _get_in_exceptions):
                raise item
            PyErr_Clear()
            return default
        coll = <object>obj
    return coll
