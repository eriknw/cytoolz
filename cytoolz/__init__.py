from .itertoolz import *

from .functoolz import *

from .dicttoolz import *

from .recipes import *

from .compatibility import map, filter

# from . import sandbox

from functools import partial, reduce

sorted = sorted

# Aliases
comp = compose

functoolz._sigs.update_signature_registry()

from ._version import __version__, __toolz_version__
