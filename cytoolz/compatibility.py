import operator
import sys
PY3 = sys.version_info[0] > 2
PY34 = sys.version_info[0] == 3 and sys.version_info[1] == 4

__all__ = ['PY3', 'map', 'filter', 'range', 'zip', 'reduce', 'zip_longest',
           'iteritems', 'iterkeys', 'itervalues', 'import_module']

map = map
filter = filter
range = range
zip = zip
from functools import reduce
from itertools import zip_longest
iteritems = operator.methodcaller('items')
iterkeys = operator.methodcaller('keys')
itervalues = operator.methodcaller('values')

from importlib import import_module
