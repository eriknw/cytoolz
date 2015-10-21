from collections import defaultdict as _defaultdict
from cytoolz.dicttoolz import (merge, merge_with, valmap, keymap, update_in,
                             assoc, dissoc, keyfilter, valfilter, itemmap,
                             itemfilter)
from cytoolz.utils import raises


class defaultdict(_defaultdict):
    def __eq__(self, other):
        return (super(defaultdict, self).__eq__(other) and
                isinstance(other, _defaultdict) and
                self.default_factory == other.default_factory)


D = dict
kw = {}


def inc(x):
    return x + 1


def iseven(i):
    return i % 2 == 0


def test_merge():
    assert merge(D({1: 1, 2: 2}), D({3: 4}), **kw) == D({1: 1, 2: 2, 3: 4})


def test_merge_iterable_arg():
    assert merge([D({1: 1, 2: 2}), D({3: 4})], **kw) == D({1: 1, 2: 2, 3: 4})


def test_merge_with():
    dicts = D({1: 1, 2: 2}), D({1: 10, 2: 20})
    assert merge_with(sum, *dicts, **kw) == D({1: 11, 2: 22})
    assert merge_with(tuple, *dicts, **kw) == D({1: (1, 10), 2: (2, 20)})

    dicts = D({1: 1, 2: 2, 3: 3}), D({1: 10, 2: 20})
    assert merge_with(sum, *dicts, **kw) == D({1: 11, 2: 22, 3: 3})
    assert merge_with(tuple, *dicts, **kw) == D({1: (1, 10), 2: (2, 20), 3: (3,)})

    assert not merge_with(sum)


def test_merge_with_iterable_arg():
    dicts = D({1: 1, 2: 2}), D({1: 10, 2: 20})
    assert merge_with(sum, *dicts, **kw) == D({1: 11, 2: 22})
    assert merge_with(sum, dicts, **kw) == D({1: 11, 2: 22})
    assert merge_with(sum, iter(dicts), **kw) == D({1: 11, 2: 22})


def test_valmap():
    assert valmap(inc, D({1: 1, 2: 2}), **kw) == D({1: 2, 2: 3})


def test_keymap():
    assert keymap(inc, D({1: 1, 2: 2}), **kw) == D({2: 1, 3: 2})


def test_itemmap():
    assert itemmap(reversed, D({1: 2, 2: 4}), **kw) == D({2: 1, 4: 2})


def test_valfilter():
    assert valfilter(iseven, D({1: 2, 2: 3}), **kw) == D({1: 2})


def test_keyfilter():
    assert keyfilter(iseven, D({1: 2, 2: 3}), **kw) == D({2: 3})


def test_itemfilter():
    assert itemfilter(lambda item: iseven(item[0]), D({1: 2, 2: 3}), **kw) == D({2: 3})
    assert itemfilter(lambda item: iseven(item[1]), D({1: 2, 2: 3}), **kw) == D({1: 2})


def test_assoc():
    assert assoc(D({}), "a", 1, **kw) == D({"a": 1})
    assert assoc(D({"a": 1}), "a", 3, **kw) == D({"a": 3})
    assert assoc(D({"a": 1}), "b", 3, **kw) == D({"a": 1, "b": 3})

    # Verify immutability:
    d = D({'x': 1})
    oldd = d
    assoc(d, 'x', 2, **kw)
    assert d is oldd


def test_dissoc():
    assert dissoc(D({"a": 1}), "a") == D({})
    assert dissoc(D({"a": 1, "b": 2}), "a") == D({"b": 2})
    assert dissoc(D({"a": 1, "b": 2}), "b") == D({"a": 1})

    # Verify immutability:
    d = D({'x': 1})
    oldd = d
    d2 = dissoc(d, 'x')
    assert d is oldd
    assert d2 is not oldd


def test_update_in():
    assert update_in(D({"a": 0}), ["a"], inc, **kw) == D({"a": 1})
    assert update_in(D({"a": 0, "b": 1}), ["b"], str, **kw) == D({"a": 0, "b": "1"})
    assert (update_in(D({"t": 1, "v": D({"a": 0})}), ["v", "a"], inc, **kw) ==
            D({"t": 1, "v": D({"a": 1})}))
    # Handle one missing key.
    assert update_in(D({}), ["z"], str, None, **kw) == D({"z": "None"})
    assert update_in(D({}), ["z"], inc, 0, **kw) == D({"z": 1})
    assert update_in(D({}), ["z"], lambda x: x+"ar", default="b", **kw) == D({"z": "bar"})
    # Same semantics as Clojure for multiple missing keys, ie. recursively
    # create nested empty dictionaries to the depth specified by the
    # keys with the innermost value set to f(default).
    assert update_in(D({}), [0, 1], inc, default=-1, **kw) == D({0: D({1: 0})})
    assert update_in(D({}), [0, 1], str, default=100, **kw) == D({0: D({1: "100"})})
    assert (update_in(D({"foo": "bar", 1: 50}), ["d", 1, 0], str, 20, **kw) ==
            D({"foo": "bar", 1: 50, "d": D({1: D({0: "20"})})}))
    # Verify immutability:
    d = D({'x': 1})
    oldd = d
    update_in(d, ['x'], inc, **kw)
    assert d is oldd


def test_factory():
    assert merge(defaultdict(int, D({1: 2})), D({2: 3})) == {1: 2, 2: 3}
    assert (merge(defaultdict(int, D({1: 2})), D({2: 3}),
                  factory=lambda: defaultdict(int)) ==
            defaultdict(int, D({1: 2, 2: 3})))
    assert not (merge(defaultdict(int, D({1: 2})), D({2: 3}),
                      factory=lambda: defaultdict(int)) == D({1: 2, 2: 3}))
    assert raises(TypeError, lambda: merge(D({1: 2}), D({2: 3}), factoryy=dict))
