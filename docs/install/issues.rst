
Installation notes
==================


Installation on Unix systems
-----------------------------

MTUQ installation is possible on Unix and Unix-like systems.


Installation on MacOS systems
-----------------------------

MTUQ installation is possible on MacOS systems.


Installation on Apple Silicon Macs
----------------------------------

MTUQ installation on Apple M1, M2, and M3 Macs is now possible using the default installation procedure.  A modified conda environment file is no longer necessary.


Installation on Windows
-----------------------

MTUQ installation is not currently supported on Windows.  We invite users to contribute portability fixes.


A common cause of installation errors on Unix systems
-----------------------------------------------------

On Unix systems, a non-writeable `/tmp` directory can cause the installation to fail in unexpected ways.  Such issues appear to originate from NumPy or conda, rather than MTUQ directly.

Since NumPy and conda appear to respect the `TMPDIR` environment variable, a useful check can be to set `TMPDIR` to a directory that exists and is writeable.


NumPy compatibility warnings
----------------------------

Following the `NumPy v2.0.0 <https://github.com/numpy/numpy/releases/tag/v2.0.0#:~:text=including%20an%20ABI%20break>`_ release, `"numpy.ndarray size changed`" and other warnings have been `widely reported <https://stackoverflow.com/a/66743692>`_.  

For the time being, we have specified ``numpy<2`` for the default installation.


Accelerating bottlenecks
------------------------

MTUQ now uses `numba.jit` for a significant speedup. The older C and Cython extension modules have been `deprecated <https://mtuqorg.github.io/mtuq/install/c_ext.html>`_.


Speeding up conda installs
--------------------------

Older versions of the conda package manager can be very slow. For a potential speedup, conda can be updated as follows:

.. code::

    conda update -n base conda

For reference, the largest potential speed up comes from the new `mamba <https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community>`_ dependency solver, which was adopted in the `23.10 release <https://conda.org/blog/2023-11-06-conda-23-10-0-release>`_.

