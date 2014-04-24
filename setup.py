""" Build `cytoolz` with or without Cython.

Deployed versions of CyToolz do not rely on Cython by default even if the
user has Cython installed.  A C compiler is used to compile the distributed
*.c files instead.

Pass "--cython" or "--with-cython" as a command line argument to setup.py
to build the project using Cython.

Pass "--no-cython" or "--without-cython" to disable usage of Cython.

For convenience, developmental versions (with 'dev' in the version number)
automatically use Cython unless disabled via a command line argument.

"""
import os.path
import sys
from distutils.core import setup
from distutils.extension import Extension

info = {}
filename = os.path.join('cytoolz', '_version.py')
exec(compile(open(filename, "rb").read(), filename, 'exec'), info)
VERSION = info['__version__']

try:
    from Cython.Distutils import build_ext
    has_cython = True
except ImportError:
    has_cython = False

is_dev = 'dev' in VERSION
use_cython = is_dev or '--cython' in sys.argv or '--with-cython' in sys.argv
if '--no-cython' in sys.argv:
    use_cython = False
    sys.argv.remove('--no-cython')
if '--without-cython' in sys.argv:
    use_cython = False
    sys.argv.remove('--without-cython')
if '--cython' in sys.argv:
    sys.argv.remove('--cython')
if '--with-cython' in sys.argv:
    sys.argv.remove('--with-cython')

if use_cython and not has_cython:
    if is_dev:
        raise RuntimeError('Cython required to build dev version of cytoolz.')
    print('WARNING: Cython not installed.  Building without Cython.')
    use_cython = False

if use_cython:
    suffix = '.pyx'
    cmdclass = {'build_ext': build_ext}
else:
    suffix = '.c'
    cmdclass = {}

ext_modules = []
for modname in ['dicttoolz', 'functoolz', 'itertoolz',
                'curried_exceptions', 'recipes']:
    ext_modules.append(Extension('cytoolz.' + modname,
                                 ['cytoolz/' + modname + suffix]))

if __name__ == '__main__':
    setup(
        name='cytoolz',
        cmdclass=cmdclass,
        ext_modules=ext_modules,
        packages=['cytoolz'],
        package_data={'cytoolz': ['*.pxd']},
    )
