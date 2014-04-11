import cytoolz
import cytoolz.curried
import toolz
import toolz.curried
import types


# Note that the tests in this file assume `toolz.curry` is a class, but we
# may some day make `toolz.curry` a function and `toolz.Curry` a class.

def test_toolzcurry_is_class():
    assert isinstance(toolz.curry, type) is True
    assert isinstance(toolz.curry, types.FunctionType) is False


def test_cytoolz_like_toolz():
    for key, val in toolz.curried.__dict__.items():
        if isinstance(val, toolz.curry):
            assert hasattr(cytoolz.curried, key), (
                    'cytoolz.curried.%s does not exist' % key)
            assert isinstance(getattr(cytoolz.curried, key), cytoolz.curry), (
                    'cytoolz.curried.%s should be curried' % key)


def test_toolz_like_cytoolz():
    for key, val in cytoolz.curried.__dict__.items():
        if isinstance(val, cytoolz.curry):
            assert hasattr(toolz.curried, key), (
                    'cytoolz.curried.%s should not exist' % key)
            assert isinstance(getattr(toolz.curried, key), toolz.curry), (
                    'cytoolz.curried.%s should not be curried' % key)
