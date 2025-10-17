
.. warning::

   With newer versions of MTUQ, the same installation procedure works for Macs
   as for other computers. Thus, the following modified procedure may no longer 
   be necessary (though it should still continue to work).



Modified installation procedure for Apple M1 and Apple M2 Macs
==============================================================

Newer Mac computers use proprietary Apple processors based on the ARM64 architecture.

To install MTUQ on ARM64 systems, create a conda virtual environment:

.. code::

   conda create -n mtuq python=3


Download the MTUQ source code:

.. code::

   git clone https://github.com/mtuqorg/mtuq.git
   cd mtuq


Install MTUQ in editable mode:

.. code::

   conda activate mtuq
   conda env update --file env_arm64.yaml


Unpack seismic waveforms used by examples:

.. code::

    bash ./data/examples/unpack.bash
    bash ./data/tests/unpack.bash


Finally, install PyGMT:

.. code::

    conda install 'pygmt=0.9'


.. note::

    Users may receive the following Instaseis warning:

    `UserWarning: Please don't install from a tarball. Use the proper pypi release or install from git.`

    In fact, env_arm64.yaml does install Instaseis from git, and the above message can be ignored.


.. note::

    If `linux-aarch64` or `osx-arm64` builds are ever added to the Instaseis `conda-forge channel <https://anaconda.org/conda-forge/instaseis>`_, the above workaround may become unnecessary.

