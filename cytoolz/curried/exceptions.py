from cytoolz import curry, merge as _merge, merge_with as _merge_with

__all__ = ['merge', 'merge_with']


@curry
def merge(d, *dicts, **kwargs):
    return _merge(d, *dicts, **kwargs)


@curry
def merge_with(func, d, *dicts, **kwargs):
    return _merge_with(func, d, *dicts, **kwargs)


merge.__doc__ = _merge.__doc__
merge_with.__doc__ = _merge_with.__doc__
