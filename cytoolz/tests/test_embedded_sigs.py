import inspect
import types
import cytoolz
import toolz


def test_class_sigs():
    """ Test that all ``cdef class`` extension types in ``cytoolz`` have
        correctly embedded the function signature as done in ``toolz``.
    """
    for key, toolz_func in sorted(toolz.__dict__.items()):
        # only consider items created in `toolz`
        toolz_mod = getattr(toolz_func, '__module__', '') or ''
        if not toolz_mod.startswith('toolz'):
            continue

        # full API coverage should be tested elsewhere
        if key not in cytoolz.__dict__:
            print 'Warning: cytoolz.%s not defined' % key
            continue

        # only test functions created in `cytoolz`
        cytoolz_func = cytoolz.__dict__[key]
        cytoolz_mod = getattr(cytoolz_func, '__module__', '') or ''
        if not cytoolz_mod.startswith('cytoolz'):
            print ('Warning: cytoolz.%s exists, but is defined outside '
                   'the package' % key)
            continue

        # only test regular Python functions or classes
        if isinstance(toolz_func, types.FunctionType):
            toolz_spec = inspect.getargspec(toolz_func)
        elif isinstance(toolz_func, type):  # class
            toolz_spec = inspect.getargspec(toolz_func.__init__)
        else:
            print ('Warning: toolz.%s is a %s, not a normal function or class'
                   % (key, type(toolz_func)))
            continue

        toolz_sig = toolz_func.__name__ + inspect.formatargspec(*toolz_spec)

        # only test `cdef class` extensions from `cytoolz`
        if isinstance(cytoolz_func, types.BuiltinFunctionType):
            continue

        if toolz_sig not in cytoolz_func.__doc__:
            message = ('cytoolz.%s does not have correct function signature.'
                       '\n\nExpected: %s'
                       '\n\nDocstring in cytoolz is:\n%s'
                       % (key, toolz_sig, cytoolz_func.__doc__))
            assert False, message
