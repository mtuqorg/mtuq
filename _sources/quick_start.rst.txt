Quick start
===========

An easy way to get started is to try out these `examples <https://github.com/mtuqorg/mtuq/blob/master/examples>`_, which estimate source parameters for an event from `Silwal2016`.

Before running any of the scripts, users must first `install MTUQ <https://mtuqorg.github.io/mtuq/install/index.html>`_.  After installation finishes, examples can be invoked within the `examples/` directory as follows:

.. code::

   python ./<name_of_example>


.. note::

  The first time an example runs, Green's functions will be downloaded, which can take several minutes.  Any subsequent runs will be much faster.


Serial example
--------------

The simplest example is `SerialGridSearch.DoubleCouple.py <https://github.com/mtuqorg/mtuq/blob/master/examples/SerialGridSearch.DoubleCouple.py>`_.  After running the script, users can check their output against this `expected result <https://github.com/rmodrak/mtbench/blob/FiguresSilwal2016/SerialGridSearch.DoubleCouple.png>`_.

A typical runtime is about 60 seconds.  For faster results, `GridSearch.DoubleCouple.py <https://github.com/mtuqorg/mtuq/blob/master/examples/GridSearch.DoubleCouple.py>`_ runs the same inversion in parallel.


Parallel examples
-----------------

All the other examples include MPI parallelization for faster evaluation on parallel systems.

To see if MPI executables and libraries are present, start by checking if `mpirun` is on the system path.  If necessary, ask a system administrator for assistance installing or configuring MPI.

If MPI is present, run the parallel examples as follows:

.. code::

   mpirun -n <nproc> python ./<name_of_example>


Note that the examples will still work if invoked without `mpirun`, but running times will be longer.


**References**

[`Silwal2016`] Vipul Silwal and Carl Tape (2016), Seismic moment tensors and
estimated uncertainties in southern Alaska, JGR, https://doi.org/10.1002/2015JB012588


