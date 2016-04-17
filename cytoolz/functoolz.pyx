import inspect
import sys
from functools import partial
from operator import attrgetter
from textwrap import dedent
from cytoolz.compatibility import PY3, PY34, filter as ifilter, map as imap, reduce
from cytoolz._signatures import (
    is_builtin_valid_args as _is_builtin_valid_args,
    is_builtin_partial_args as _is_builtin_partial_args,
    has_unknown_args as _has_unknown_args,
    signature_or_spec as _signature_or_spec,
)
from cpython.dict cimport PyDict_Merge, PyDict_New
from cpython.exc cimport PyErr_Clear, PyErr_Occurred, PyErr_GivenExceptionMatches
from cpython.object cimport (PyCallable_Check, PyObject_Call, PyObject_CallObject,
                             PyObject_RichCompare, Py_EQ, Py_NE)
from cpython.ref cimport PyObject, Py_DECREF
from cpython.sequence cimport PySequence_Concat
from cpython.set cimport PyFrozenSet_New
from cpython.tuple cimport PyTuple_Check, PyTuple_GET_SIZE

# Locally defined bindings that differ from `cython.cpython` bindings
from cytoolz.cpython cimport PtrObject_Call


__all__ = ['identity', 'thread_first', 'thread_last', 'memoize', 'compose',
           'pipe', 'complement', 'juxt', 'do', 'curry', 'memoize', 'flip',
           'excepts']


cpdef object identity(object x):
    return x


cdef object c_thread_first(object val, object forms):
    cdef object form, func
    cdef tuple args
    for form in forms:
        if PyCallable_Check(form):
            val = form(val)
        elif PyTuple_Check(form):
            func, args = form[0], (val,) + form[1:]
            val = PyObject_CallObject(func, args)
        else:
            val = None
    return val


def thread_first(val, *forms):
    """
    Thread value through a sequence of functions/forms

    >>> def double(x): return 2*x
    >>> def inc(x):    return x + 1
    >>> thread_first(1, inc, double)
    4

    If the function expects more than one input you can specify those inputs
    in a tuple.  The value is used as the first input.

    >>> def add(x, y): return x + y
    >>> def pow(x, y): return x**y
    >>> thread_first(1, (add, 4), (pow, 2))  # pow(add(1, 4), 2)
    25

    So in general
        thread_first(x, f, (g, y, z))
    expands to
        g(f(x), y, z)

    See Also:
        thread_last
    """
    return c_thread_first(val, forms)


cdef object c_thread_last(object val, object forms):
    cdef object form, func
    cdef tuple args
    for form in forms:
        if PyCallable_Check(form):
            val = form(val)
        elif PyTuple_Check(form):
            func, args = form[0], form[1:] + (val,)
            val = PyObject_CallObject(func, args)
        else:
            val = None
    return val


def thread_last(val, *forms):
    """
    Thread value through a sequence of functions/forms

    >>> def double(x): return 2*x
    >>> def inc(x):    return x + 1
    >>> thread_last(1, inc, double)
    4

    If the function expects more than one input you can specify those inputs
    in a tuple.  The value is used as the last input.

    >>> def add(x, y): return x + y
    >>> def pow(x, y): return x**y
    >>> thread_last(1, (add, 4), (pow, 2))  # pow(2, add(4, 1))
    32

    So in general
        thread_last(x, f, (g, y, z))
    expands to
        g(y, z, f(x))

    >>> def iseven(x):
    ...     return x % 2 == 0
    >>> list(thread_last([1, 2, 3], (map, inc), (filter, iseven)))
    [2, 4]

    See Also:
        thread_first
    """
    return c_thread_last(val, forms)


cdef struct partialobject:
    PyObject _
    PyObject *fn
    PyObject *args
    PyObject *kw
    PyObject *dict
    PyObject *weakreflist


cdef object _partial = partial(lambda: None)


cdef object _empty_kwargs():
    if <object> (<partialobject*> _partial).kw is None:
        return None
    return PyDict_New()


cdef class curry:
    """ curry(self, *args, **kwargs)

    Curry a callable function

    Enables partial application of arguments through calling a function with an
    incomplete set of arguments.

    >>> def mul(x, y):
    ...     return x * y
    >>> mul = curry(mul)

    >>> double = mul(2)
    >>> double(10)
    20

    Also supports keyword arguments

    >>> @curry                  # Can use curry as a decorator
    ... def f(x, y, a=10):
    ...     return a * (x + y)

    >>> add = f(a=1)
    >>> add(2, 3)
    5

    See Also:
        cytoolz.curried - namespace of curried functions
                        http://toolz.readthedocs.org/en/latest/curry.html
    """
    property __wrapped__:
        def __get__(self):
            return self.func

    def __cinit__(self, *args, **kwargs):
        if not args:
            raise TypeError('__init__() takes at least 2 arguments (1 given)')
        func, args = args[0], args[1:]
        if not PyCallable_Check(func):
            raise TypeError("Input must be callable")

        # curry- or functools.partial-like object?  Unpack and merge arguments
        if (hasattr(func, 'func')
                and hasattr(func, 'args')
                and hasattr(func, 'keywords')
                and isinstance(func.args, tuple)):
            if func.keywords:
                PyDict_Merge(kwargs, func.keywords, False)
                ## Equivalent to:
                # for key, val in func.keywords.items():
                #     if key not in kwargs:
                #         kwargs[key] = val
            args = func.args + args
            func = func.func

        self.func = func
        self.args = args
        self.keywords = kwargs if kwargs else _empty_kwargs()
        self.__doc__ = getattr(func, '__doc__', None)
        self.__name__ = getattr(func, '__name__', '<curry>')
        self._sigspec = None
        self._has_unknown_args = None

    def __str__(self):
        return str(self.func)

    def __repr__(self):
        return repr(self.func)

    def __hash__(self):
        return hash((self.func, self.args,
                     frozenset(self.keywords.items()) if self.keywords
                     else None))

    def __richcmp__(self, other, int op):
        is_equal = (isinstance(other, curry) and self.func == other.func and
                self.args == other.args and self.keywords == other.keywords)
        if op == Py_EQ:
            return is_equal
        if op == Py_NE:
            return not is_equal
        return PyObject_RichCompare(id(self), id(other), op)

    def __call__(self, *args, **kwargs):
        cdef PyObject *obj
        cdef object val

        if PyTuple_GET_SIZE(args) == 0:
            args = self.args
        elif PyTuple_GET_SIZE(self.args) != 0:
            args = PySequence_Concat(self.args, args)
        if self.keywords is not None:
            PyDict_Merge(kwargs, self.keywords, False)

        obj = PtrObject_Call(self.func, args, kwargs)
        if obj is not NULL:
            val = <object>obj
            Py_DECREF(val)
            return val

        val = <object>PyErr_Occurred()
        PyErr_Clear()
        if (PyErr_GivenExceptionMatches(val, TypeError) and
            self._should_curry_internal(args, kwargs, val)
        ):
            return type(self)(self.func, *args, **kwargs)
        raise val

    def _should_curry_internal(self, args, kwargs, exc=None):
        func = self.func

        # `toolz` has these three lines
        #args = self.args + args
        #if self.keywords:
        #    kwargs = dict(self.keywords, **kwargs)

        if self._sigspec is None:
            sigspec = self._sigspec = _signature_or_spec(func)
            self._has_unknown_args = _has_unknown_args(func, sigspec=sigspec)
        else:
            sigspec = self._sigspec

        if is_partial_args(func, args, kwargs, sigspec=sigspec) is False:
            # Nothing can make the call valid
            return False
        elif self._has_unknown_args:
            # The call may be valid and raised a TypeError, but we curry
            # anyway because the function may have `*args`.  This is useful
            # for decorators with signature `func(*args, **kwargs)`.
            return True
        elif not is_valid_args(func, args, kwargs, sigspec=sigspec):
            # Adding more arguments may make the call valid
            return True
        else:
            # There was a genuine TypeError
            return False

    def bind(self, *args, **kwargs):
        return type(self)(self, *args, **kwargs)

    def call(self, *args, **kwargs):
        cdef PyObject *obj
        cdef object val

        if PyTuple_GET_SIZE(args) == 0:
            args = self.args
        elif PyTuple_GET_SIZE(self.args) != 0:
            args = PySequence_Concat(self.args, args)
        if self.keywords is not None:
            PyDict_Merge(kwargs, self.keywords, False)

        obj = PtrObject_Call(self.func, args, kwargs)
        if obj is not NULL:
            val = <object>obj
            Py_DECREF(val)
            return val

        val = <object>PyErr_Occurred()
        PyErr_Clear()
        raise val

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return type(self)(self, instance)

    def __reduce__(self):
        return (type(self), (self.func,), (self.args, self.keywords))

    def __setstate__(self, state):
        self.args, self.keywords = state


cpdef object has_kwargs(object f):
    """
    Does a function have keyword arguments?

    >>> def f(x, y=0):
    ...     return x + y

    >>> has_kwargs(f)
    True
    """
    if sys.version_info[0] == 2:
        spec = inspect.getargspec(f)
        return bool(spec and (spec.keywords or spec.defaults))
    if sys.version_info[0] == 3:
        spec = inspect.getfullargspec(f)
        return bool(spec.defaults)


cpdef object isunary(object f):
    """
    Does a function have only a single argument?

    >>> def f(x):
    ...     return x

    >>> isunary(f)
    True
    >>> isunary(lambda x, y: x + y)
    False
    """
    cdef int major = sys.version_info[0]
    try:
        if major == 2:
            spec = inspect.getargspec(f)
        if major == 3:
            spec = inspect.getfullargspec(f)
        return bool(spec and spec.varargs is None and not has_kwargs(f)
                    and len(spec.args) == 1)
    except TypeError:
        pass
    return None    # in Python < 3.4 builtins fail, return None


cdef class c_memoize:
    property __doc__:
        def __get__(self):
            return self.func.__doc__

    property __name__:
        def __get__(self):
            return self.func.__name__

    property __wrapped__:
        def __get__(self):
            return self.func

    def __cinit__(self, func, cache=None, key=None):
        self.func = func
        if cache is None:
            self.cache = PyDict_New()
        else:
            self.cache = cache
        self.key = key

        try:
            self.may_have_kwargs = has_kwargs(func)
            # Is unary function (single arg, no variadic argument or keywords)?
            self.is_unary = isunary(func)
        except TypeError:
            self.is_unary = False
            self.may_have_kwargs = True

    def __call__(self, *args, **kwargs):
        cdef object key
        if self.key is not None:
            key = self.key(args, kwargs)
        elif self.is_unary:
            key = args[0]
        elif self.may_have_kwargs:
            key = (args or None,
                   PyFrozenSet_New(kwargs.items()) if kwargs else None)
        else:
            key = args

        if key in self.cache:
            return self.cache[key]
        else:
            result = PyObject_Call(self.func, args, kwargs)
            self.cache[key] = result
            return result

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return curry(self, instance)


cpdef object memoize(object func=None, object cache=None, object key=None):
    """
    Cache a function's result for speedy future evaluation

    Considerations:
        Trades memory for speed.
        Only use on pure functions.

    >>> def add(x, y):  return x + y
    >>> add = memoize(add)

    Or use as a decorator

    >>> @memoize
    ... def add(x, y):
    ...     return x + y

    Use the ``cache`` keyword to provide a dict-like object as an initial cache

    >>> @memoize(cache={(1, 2): 3})
    ... def add(x, y):
    ...     return x + y

    Note that the above works as a decorator because ``memoize`` is curried.

    It is also possible to provide a ``key(args, kwargs)`` function that
    calculates keys used for the cache, which receives an ``args`` tuple and
    ``kwargs`` dict as input, and must return a hashable value.  However,
    the default key function should be sufficient most of the time.

    >>> # Use key function that ignores extraneous keyword arguments
    >>> @memoize(key=lambda args, kwargs: args)
    ... def add(x, y, verbose=False):
    ...     if verbose:
    ...         print('Calculating %s + %s' % (x, y))
    ...     return x + y
    """
    # pseudo-curry
    if func is None:
        return curry(c_memoize, cache=cache, key=key)
    return c_memoize(func, cache=cache, key=key)


cdef class Compose:
    """ Compose(self, *funcs)

    A composition of functions

    See Also:
        compose
    """
    def __cinit__(self, *funcs):
        self.first = funcs[-1]
        self.funcs = tuple(reversed(funcs[:-1]))

    def __call__(self, *args, **kwargs):
        cdef object func, ret
        ret = PyObject_Call(self.first, args, kwargs)
        for func in self.funcs:
            ret = func(ret)
        return ret

    def __reduce__(self):
        return (Compose, (self.first,), self.funcs)

    def __setstate__(self, state):
        self.funcs = state

    property __name__:
        def __get__(self):
            try:
                return '_of_'.join(
                    f.__name__ for f in reversed((self.first,) + self.funcs)
                )
            except AttributeError:
                return type(self).__name__

    property __doc__:
        def __get__(self):
            def composed_doc(*fs):
                """Generate a docstring for the composition of fs.
                """
                if not fs:
                    # Argument name for the docstring.
                    return '*args, **kwargs'

                return '{f}({g})'.format(f=fs[0].__name__, g=composed_doc(*fs[1:]))

            try:
                return (
                    'lambda *args, **kwargs: ' +
                    composed_doc(*reversed((self.first,) + self.funcs))
                )
            except AttributeError:
                # One of our callables does not have a `__name__`, whatever.
                return 'A composition of functions'


cdef object c_compose(object funcs):
    if not funcs:
        return identity
    elif len(funcs) == 1:
        return funcs[0]
    else:
        return Compose(*funcs)


def compose(*funcs):
    """
    Compose functions to operate in series.

    Returns a function that applies other functions in sequence.

    Functions are applied from right to left so that
    ``compose(f, g, h)(x, y)`` is the same as ``f(g(h(x, y)))``.

    If no arguments are provided, the identity function (f(x) = x) is returned.

    >>> inc = lambda i: i + 1
    >>> compose(str, inc)(3)
    '4'

    See Also:
        pipe
    """
    return c_compose(funcs)


cdef object c_pipe(object data, object funcs):
    cdef object func
    for func in funcs:
        data = func(data)
    return data


def pipe(data, *funcs):
    """
    Pipe a value through a sequence of functions

    I.e. ``pipe(data, f, g, h)`` is equivalent to ``h(g(f(data)))``

    We think of the value as progressing through a pipe of several
    transformations, much like pipes in UNIX

    ``$ cat data | f | g | h``

    >>> double = lambda i: 2 * i
    >>> pipe(3, double, str)
    '6'

    See Also:
        compose
        thread_first
        thread_last
    """
    return c_pipe(data, funcs)


cdef class complement:
    """ complement(func)

    Convert a predicate function to its logical complement.

    In other words, return a function that, for inputs that normally
    yield True, yields False, and vice-versa.

    >>> def iseven(n): return n % 2 == 0
    >>> isodd = complement(iseven)
    >>> iseven(2)
    True
    >>> isodd(2)
    False
    """
    def __cinit__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return not PyObject_Call(self.func, args, kwargs)  # use PyObject_Not?

    def __reduce__(self):
        return (complement, (self.func,))


cdef class _juxt_inner:
    def __cinit__(self, funcs):
        self.funcs = tuple(funcs)

    def __call__(self, *args, **kwargs):
        if kwargs:
            return tuple(PyObject_Call(func, args, kwargs) for func in self.funcs)
        else:
            return tuple(PyObject_CallObject(func, args) for func in self.funcs)

    def __reduce__(self):
        return (_juxt_inner, (self.funcs,))


cdef object c_juxt(object funcs):
    return _juxt_inner(funcs)


def juxt(*funcs):
    """
    Creates a function that calls several functions with the same arguments.

    Takes several functions and returns a function that applies its arguments
    to each of those functions then returns a tuple of the results.

    Name comes from juxtaposition: the fact of two things being seen or placed
    close together with contrasting effect.

    >>> inc = lambda x: x + 1
    >>> double = lambda x: x * 2
    >>> juxt(inc, double)(10)
    (11, 20)
    >>> juxt([inc, double])(10)
    (11, 20)
    """
    if len(funcs) == 1 and not PyCallable_Check(funcs[0]):
        funcs = funcs[0]
    return c_juxt(funcs)


cpdef object do(object func, object x):
    """
    Runs ``func`` on ``x``, returns ``x``

    Because the results of ``func`` are not returned, only the side
    effects of ``func`` are relevant.

    Logging functions can be made by composing ``do`` with a storage function
    like ``list.append`` or ``file.write``

    >>> from cytoolz import compose
    >>> from cytoolz.curried import do

    >>> log = []
    >>> inc = lambda x: x + 1
    >>> inc = compose(inc, do(log.append))
    >>> inc(1)
    2
    >>> inc(11)
    12
    >>> log
    [1, 11]

    """
    func(x)
    return x


cpdef object _flip(object f, object a, object b):
    return PyObject_CallObject(f, (b, a))


flip = curry(_flip)


cpdef object return_none(object exc):
    """
    Returns None.
    """
    return None


cdef class excepts:
    """
    A wrapper around a function to catch exceptions and
    dispatch to a handler.

    This is like a functional try/except block, in the same way that
    ifexprs are functional if/else blocks.

    Examples
    --------
    >>> excepting = excepts(
    ...     ValueError,
    ...     lambda a: [1, 2].index(a),
    ...     lambda _: -1,
    ... )
    >>> excepting(1)
    0
    >>> excepting(3)
    -1

    Multiple exceptions and default except clause.
    >>> excepting = excepts((IndexError, KeyError), lambda a: a[0])
    >>> excepting([])
    >>> excepting([1])
    1
    >>> excepting({})
    >>> excepting({0: 1})
    1
    """

    def __init__(self, exc, func, handler=return_none):
        self.exc = exc
        self.func = func
        self.handler = handler

    def __call__(self, *args, **kwargs):
        try:
            return self.func(*args, **kwargs)
        except self.exc as e:
            return self.handler(e)

    property __name__:
        def __get__(self):
            exc = self.exc
            try:
                if isinstance(exc, tuple):
                    exc_name = '_or_'.join(map(attrgetter('__name__'), exc))
                else:
                    exc_name = exc.__name__
                return '%s_excepting_%s' % (self.func.__name__, exc_name)
            except AttributeError:
                return 'excepting'

    property __doc__:
        def __get__(self):
            exc = self.exc
            try:
                if isinstance(exc, tuple):
                    exc_name = '(%s)' % ', '.join(
                        map(attrgetter('__name__'), exc),
                    )
                else:
                    exc_name = exc.__name__

                return dedent(
                    """\
                    A wrapper around {inst.func.__name__!r} that will except:
                    {exc}
                    and handle any exceptions with {inst.handler.__name__!r}.

                    Docs for {inst.func.__name__!r}:
                    {inst.func.__doc__}

                    Docs for {inst.handler.__name__!r}:
                    {inst.handler.__doc__}
                    """
                ).format(
                    inst=self,
                    exc=exc_name,
                )
            except AttributeError:
                return type(self).__doc__


cpdef object is_valid_args(object func, object args, object kwargs, object sigspec=None):
    if PY34:
        val = _is_builtin_valid_args(func, args, kwargs)
        if val is not None:
            return val
    if PY3:
        if sigspec is None:
            try:
                sigspec = inspect.signature(func)
            except (ValueError, TypeError) as e:
                sigspec = e
        if isinstance(sigspec, ValueError):
            return _is_builtin_valid_args(func, args, kwargs)
        elif isinstance(sigspec, TypeError):
            return False
        try:
            sigspec.bind(*args, **kwargs)
        except (TypeError, AttributeError):
            return False
        return True

    else:
        if sigspec is None:
            try:
                sigspec = inspect.getargspec(func)
            except TypeError as e:
                sigspec = e
        if isinstance(sigspec, TypeError):
            if not callable(func):
                return False
            return _is_builtin_valid_args(func, args, kwargs)

        spec = sigspec
        defaults = spec.defaults or ()
        num_pos = len(spec.args) - len(defaults)
        missing_pos = spec.args[len(args):num_pos]
        for arg in missing_pos:
            if arg not in kwargs:
                return False

        if spec.varargs is None:
            num_extra_pos = max(0, len(args) - num_pos)
        else:
            num_extra_pos = 0

        kwargs = dict(kwargs)

        # Add missing keyword arguments (unless already included in `args`)
        missing_kwargs = spec.args[num_pos + num_extra_pos:]
        kwargs.update(zip(missing_kwargs, defaults[num_extra_pos:]))

        # Convert call to use positional arguments
        more_args = []
        for key in spec.args[len(args):]:
            more_args.append(kwargs.pop(key))
        args = args + tuple(more_args)

        if (
            not spec.keywords and kwargs or
            not spec.varargs and len(args) > len(spec.args) or
            set(spec.args[:len(args)]) & set(kwargs)
        ):
            return False
        else:
            return True


cpdef object is_partial_args(object func, object args, object kwargs, object sigspec=None):
    if PY34:
        val = _is_builtin_partial_args(func, args, kwargs)
        if val is not None:
            return val
    if PY3:
        if sigspec is None:
            try:
                sigspec = inspect.signature(func)
            except (ValueError, TypeError) as e:
                sigspec = e
        if isinstance(sigspec, ValueError):
            return _is_builtin_partial_args(func, args, kwargs)
        elif isinstance(sigspec, TypeError):
            return False
        try:
            sigspec.bind_partial(*args, **kwargs)
        except (TypeError, AttributeError):
            return False
        return True

    else:
        if sigspec is None:
            try:
                sigspec = inspect.getargspec(func)
            except TypeError as e:
                sigspec = e
        if isinstance(sigspec, TypeError):
            if not callable(func):
                return False
            return _is_builtin_partial_args(func, args, kwargs)

        spec = sigspec
        defaults = spec.defaults or ()
        num_pos = len(spec.args) - len(defaults)
        if spec.varargs is None:
            num_extra_pos = max(0, len(args) - num_pos)
        else:
            num_extra_pos = 0

        kwargs = dict(kwargs)

        # Add missing keyword arguments (unless already included in `args`)
        missing_kwargs = spec.args[num_pos + num_extra_pos:]
        kwargs.update(zip(missing_kwargs, defaults[num_extra_pos:]))

        # Add missing position arguments as keywords (may already be in kwargs)
        missing_args = spec.args[len(args):num_pos + num_extra_pos]
        for x in missing_args:
            kwargs[x] = None

        # Convert call to use positional arguments
        more_args = []
        for key in spec.args[len(args):]:
            more_args.append(kwargs.pop(key))
        args = args + tuple(more_args)

        if (
            not spec.keywords and kwargs or
            not spec.varargs and len(args) > len(spec.args) or
            set(spec.args[:len(args)]) & set(kwargs)
        ):
            return False
        else:
            return True

