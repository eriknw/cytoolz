import inspect
import cytoolz
import toolz

from types import BuiltinFunctionType
from cytoolz import curry, identity, keyfilter, valfilter, merge_with


@curry
def isfrommod(modname, func):
    mod = getattr(func, '__module__', '') or ''
    return modname in mod


def test_class_sigs():
    """ Test that all ``cdef class`` extension types in ``cytoolz`` have
        correctly embedded the function signature as done in ``toolz``.
    """
    # only consider items created in both `toolz` and `cytoolz`
    toolz_dict = valfilter(isfrommod('toolz'), toolz.__dict__)
    cytoolz_dict = valfilter(isfrommod('cytoolz'), cytoolz.__dict__)

    # only test `cdef class` extensions from `cytoolz`
    cytoolz_dict = valfilter(lambda x: not isinstance(x, BuiltinFunctionType),
                             cytoolz_dict)

    # full API coverage should be tested elsewhere
    toolz_dict = keyfilter(cytoolz_dict.has_key, toolz_dict)
    cytoolz_dict = keyfilter(toolz_dict.has_key, cytoolz_dict)

    d = merge_with(identity, toolz_dict, cytoolz_dict)
    for key, (toolz_func, cytoolz_func) in d.items():
        try:
            # function
            toolz_spec = inspect.getargspec(toolz_func)
        except TypeError:
            try:
                # curried or partial object
                toolz_spec = inspect.getargspec(toolz_func.func)
            except (TypeError, AttributeError):
                # class
                toolz_spec = inspect.getargspec(toolz_func.__init__)

        toolz_sig = toolz_func.__name__ + inspect.formatargspec(*toolz_spec)
        if toolz_sig not in cytoolz_func.__doc__:
            message = ('cytoolz.%s does not have correct function signature.'
                       '\n\nExpected: %s'
                       '\n\nDocstring in cytoolz is:\n%s'
                       % (key, toolz_sig, cytoolz_func.__doc__))
            assert False, message
