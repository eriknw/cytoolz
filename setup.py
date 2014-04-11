from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension("cytoolz.itertoolz",
              ["cytoolz/itertoolz.pyx"]),
    Extension("cytoolz.functoolz",
              ["cytoolz/functoolz.pyx"]),
    Extension("cytoolz.dicttoolz",
              ["cytoolz/dicttoolz.pyx"]),
    Extension("cytoolz.recipes",
              ["cytoolz/recipes.pyx"]),
    Extension("cytoolz.curried_exceptions",
              ["cytoolz/curried_exceptions.pyx"]),
]

setup(
    name="cytoolz",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    packages=['cytoolz'],
    package_data={'cytoolz': ['*.pxd']},
)
