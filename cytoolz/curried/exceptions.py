import cytoolz
#from cytoolz import curry, merge as _merge, merge_with as _merge_with

__all__ = ['merge', 'merge_with']


@cytoolz.curry
def merge(d, *dicts, **kwargs):
    return cytoolz.merge(d, *dicts, **kwargs)


@cytoolz.curry
def merge_with(func, d, *dicts, **kwargs):
    return cytoolz.merge_with(func, d, *dicts, **kwargs)


merge.__doc__ = cytoolz.merge.__doc__
merge_with.__doc__ = cytoolz.merge_with.__doc__
