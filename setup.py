from distutils.core import setup
from distutils.extension import Extension
# from Cython.Build import cythonize
from Cython.Distutils import build_ext

ext_modules = [
    Extension("coolz.itertoolz.core",
              ["coolz/itertoolz/core.pyx"]),
    Extension("coolz.functoolz.core",
              ["coolz/functoolz/core.pyx"]),
    Extension("coolz.dicttoolz.core",
              ["coolz/dicttoolz/core.pyx"]),
]

setup(
    name="coolz",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    packages=['coolz',
              'coolz.itertoolz',
              'coolz.functoolz',
              'coolz.dicttoolz',
              # 'coolz.sandbox',
              ],
    package_data={'coolz': ['*.pxd', '*/*.pxd']},
)
