Installation
============

We recommend installing MTUQ under `Miniforge <https://conda-forge.org/download/>`_.


To install MTUQ, first download the source code:

.. code::

   git clone https://github.com/mtuqorg/mtuq.git
   cd mtuq


Create a conda virtual environment:

.. code::

   conda create --name=mtuq
   conda activate mtuq


Then install in editable mode:

.. code::

   conda env update --name=mtuq --file=env.yaml


Unpack seismic waveforms used by examples:

.. code::

    bash ./data/examples/unpack.bash
    bash ./data/tests/download.bash


**Troubleshooting**

For troubleshooting common installation issues, please see `

- `general troubleshooting <https://mtuqorg.github.io/mtuq/install/issues.html>`_

- `graphics troubleshooting <https://mtuqorg.github.io/mtuq/install/graphics.html>`_



