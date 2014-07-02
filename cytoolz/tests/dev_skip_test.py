import cytoolz
from nose.tools import nottest, istest

# Decorator used to skip tests for developmental versions of CyToolz
if 'dev' in cytoolz.__version__:
    dev_skip_test = nottest
else:
    dev_skip_test = istest
