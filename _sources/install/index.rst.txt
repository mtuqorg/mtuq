Installation
============

We recommend installing MTUQ under Miniforge, which can be obtained following `these instructions <https://conda-forge.org/download/>`_.


To install MTUQ, create a conda virtual environment:

.. code::

   conda create -n mtuq


Download the MTUQ source code:

.. code::

   git clone https://github.com/mtuqorg/mtuq.git
   cd mtuq


Install in editable mode:

.. code::

   conda activate mtuq
   conda env update --file env.yaml


Unpack seismic waveforms used by examples:

.. code::

    bash ./data/examples/unpack.bash
    bash ./data/tests/download.bash


**Troubleshooting**

For troubleshooting common installation issues, please see `

- `general troubleshooting <https://mtuqorg.github.io/mtuq/install/issues.html>`_

- `graphics troubleshooting <https://mtuqorg.github.io/mtuq/install/graphics.html>`_



