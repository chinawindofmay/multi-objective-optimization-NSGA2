import setuptools
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("nsga2_0817.py")
)
