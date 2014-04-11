"""
Alternate namespece for cytoolz such that all functions are curried

Currying provides implicit partial evaluation of all functions

Example:

    Get usually requires two arguments, an index and a collection
    >>> from cytoolz.curried import get
    >>> get(0, ('a', 'b'))
    'a'

    When we use it in higher order functions we often want to pass a partially
    evaluated form
    >>> data = [(1, 2), (11, 22), (111, 222)]
    >>> list(map(lambda seq: get(0, seq), data))
    [1, 11, 111]

    The curried version allows simple expression of partial evaluation
    >>> list(map(get(0), data))
    [1, 11, 111]

See Also:
    cytoolz.functoolz.curry
"""

import cytoolz
from cytoolz import *
from cytoolz.curried_exceptions import *


# Here is the recipe used to create the list below
# (and "cytoolz/tests/test_curried_toolzlike.py" verifies the list is correct):
#
# import toolz
# import toolz.curried
#
# for item in sorted(key for key, val in toolz.curried.__dict__.items()
#                    if isinstance(val, toolz.curry)):
#      print '%s = cytoolz.curry(%s)' % (item, item)

accumulate = cytoolz.curry(accumulate)
assoc = cytoolz.curry(assoc)
cons = cytoolz.curry(cons)
countby = cytoolz.curry(countby)
do = cytoolz.curry(do)
drop = cytoolz.curry(drop)
filter = cytoolz.curry(filter)
get = cytoolz.curry(get)
get_in = cytoolz.curry(get_in)
groupby = cytoolz.curry(groupby)
interleave = cytoolz.curry(interleave)
interpose = cytoolz.curry(interpose)
iterate = cytoolz.curry(iterate)
keyfilter = cytoolz.curry(keyfilter)
keymap = cytoolz.curry(keymap)
map = cytoolz.curry(map)
mapcat = cytoolz.curry(mapcat)
memoize = cytoolz.curry(memoize)
merge_with = cytoolz.curry(merge_with)
nth = cytoolz.curry(nth)
partition = cytoolz.curry(partition)
partition_all = cytoolz.curry(partition_all)
partitionby = cytoolz.curry(partitionby)
pluck = cytoolz.curry(pluck)
reduce = cytoolz.curry(reduce)
reduceby = cytoolz.curry(reduceby)
remove = cytoolz.curry(remove)
sliding_window = cytoolz.curry(sliding_window)
sorted = cytoolz.curry(sorted)
take = cytoolz.curry(take)
take_nth = cytoolz.curry(take_nth)
unique = cytoolz.curry(unique)
update_in = cytoolz.curry(update_in)
valfilter = cytoolz.curry(valfilter)
valmap = cytoolz.curry(valmap)
