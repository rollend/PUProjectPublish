from setuptools import setup
from Cython.Build import cythonize

setup(
    name='patientKG',
    ext_modules=cythonize("graphs_base.py"),
    zip_safe=False,
)