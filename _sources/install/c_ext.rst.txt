
.. warning::

    As of October 2025, MTUQ's C and Cython extensions are still present in the source code, but no longer active following the default conda or pip installations.


.. warning::

    As of July 2025,  MTUQ's C and Cython extensions are deprecated and no longer maintained.


Extension modules
-----------------

MTUQ includes C and Cython extension modules for accelerating computational bottlenecks.


Compiling Cython extensions
---------------------------

The Cython extension modules can be compiled using the `c-compiler  <https://anaconda.org/conda-forge/compilers>`_ conda-forge package, which seems to work well on most Linux, Mac, and Windows systems.

.. code::

    conda install c-compiler

To compile the Cython extensions:

.. code::

    ./build_ext.sh


To troubleshoot Cython installation, users can try modifying the Cython source code files given in `setup.py`, or try varying the compilation settings determined by `setup.py`, the `CC` environment variable, the conda environment, and the underlying system environment.



