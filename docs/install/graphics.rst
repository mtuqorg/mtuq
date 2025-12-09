
Graphics notes
==============


Frontends versus backends
-------------------------

Under the hood, MTUQ plotting functions are separated into `frontends <https://mtuqorg.github.io/mtuq/library/index.html#data-visualization>`_, which users can invoke directly, and backends, which isolate calls to graphics libraries.

In addition to helping with planned transitions from one graphics library to another, this separation helps with graphics `customizability <https://mtuqorg.github.io/mtuq/user_guide/06/customizing_figures.html>`_.


matplotlib backends
-------------------

By default, MTUQ uses matplotlib backends.


GMT and PyGMT backends
----------------------

In addition to newer matplotlib backends, MTUQ includes older GMT and PyGMT backends.  To use these, it is necessary to install PyGMT and then supply the backend function when invoking the frontend.


PyGMT installation
------------------

To avoid known issues with early PyGMT versions, it is necessary to specify a minimum version as follows:

.. code::

    conda install 'pygmt>=0.9'


A conservative approach, which avoids possible issues with PyGMT 0.10 or later releases, is to specify the stable PyGMT 0.9 version:

.. code::

    conda install 'pygmt=0.9'



Moment tensor beachballs
------------------------

MTUQ now includes its own beachball plotting implementation, as well as wrappers over GMT, PyGMT and ObsPy beachball plotting functions.


.. warning::

    Some versions of GMT and ObsPy plotted `non-double couple <https://github.com/obspy/obspy/issues/2388>`_ beachballs incorrectly.

.. warning::

    MTUQ's native `beachball plotting <https://mtuqorg.github.io/mtuq/_modules/mtuq/graphics/beachball.html>`_ appears to improve over some versions of GMT and ObsPy, but is still being tested.

