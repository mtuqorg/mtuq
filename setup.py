
import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

setup_args = {
    'name': 'mtuq',
    'cmdclass': {'build_ext': build_ext}
    }

try:
    import numpy
    from Cython.Build import cythonize

    ext_modules=cythonize(
        "mtuq/misfit/waveform/ext_cython.pyx",
        compiler_directives={'language_level': "3"},
        include_dirs=[numpy.get_include()],
        )

    setup(ext_modules=ext_modules, **setup_args)

except:
    # retry without Cython extensions
    if 'build_ext' in setup_args['cmdclass']:
        del setup_args['cmdclass']['build_ext']

    setup(**setup_args)

