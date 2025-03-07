
Installation notes
==================


Installation on Apple Silicon Macs
----------------------------------

MTUQ installation on Apple M1, M2 and M3 Macs is now possible using the default installation procedure.  A modified conda environment file is no longer necessary.


Cython extension modules
------------------------

MTUQ uses Cython extension modules for a significant speed up.

By default, Cython extension modules are compiled using the conda-forge `c-compiler  <https://anaconda.org/conda-forge/compilers>`_ package, which seems to work well on most Linux, Mac, and Windows systems.


Cython compilation settings
---------------------------

To experiment with different compilation settings, users are free to comment out `c-compiler` in `env.yaml` and specify a different compiler via the `CC` environment vairable.

For fast experimentation, users can run 

.. code::

    ./build_ext.sh

which compiles the Cython extensions without first installing other dependencies in the conda environment file.


Troubleshooting Cython compilation
----------------------------------

If Cython modules fail to compile, MTUQ installation will exit with an error message ending with `CondaEnvException: Pip failed`.  More informative output can then be obtained by running `build_ext.sh`.

To troubleshoot Cython installation, users can try modifying the Cython source code files listed in `setup.py`, or try varying the compilation settings determined by `setup.py`, the `CC` environment variable, the conda environment, and the underlying system environment.

Alternatively, users can bypass Cython compilation errors by adding `optional=True` to the extension module settings in `setup.py`:

.. code::

    ext_modules = [
        Extension(
            'mtuq.misfit.waveform.c_ext_L2', ['mtuq/misfit/waveform/c_ext_L2.c'],
            include_dirs=[numpy.get_include()],
            extra_compile_args=get_compile_args(),
            optional=True,
        )

With this change, Cython errors will be ignored at install-time and MTUQ may fall back to slow pure Python functions at runtime.



Instaseis installation
----------------------

MTUQ uses Instaseis to generate synthetic seismograms.

Because Instaseis does not always install successfully under conda-forge, we implement a workaround using a modified Instaseis repository hosted on GitHub.

Similar to Cython extensions, Instaseis uses Fortran extensions for speedup, for which we install `fortran-compiler` conda-forge package.



Speeding up conda installs
--------------------------

Older versions of the conda package manager can be very slow. For a potential speed up, conda can be updated as follows:

.. code::

    conda update -n base conda

For reference, the largest potential speed up comes from the new `mamba <https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community>`_ dependency solver, which was `adopted <https://conda.org/blog/2023-11-06-conda-23-10-0-release>`_ in the 23.10 release.


