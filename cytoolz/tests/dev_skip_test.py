import nose.tools
import cytoolz

# Decorator used to skip tests for developmental versions of CyToolz
if 'dev' in cytoolz.__version__:
    dev_skip_test = nose.tools.nottest
else:
    dev_skip_test = nose.tools.istest
