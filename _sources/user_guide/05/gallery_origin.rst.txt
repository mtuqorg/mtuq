

Plotting depth and hypocenter results
-------------------------------------

Suppose we are running a grid search over both `origins` and moment tensor `sources`:

.. code::

    ds = grid_search(data, greens, misfit, origins, sources)


After the above command finishes, the data structure `ds` will contain all the moment tensors and corresponding misfit values.


Plotting misfit versus depth
""""""""""""""""""""""""""""

If `origins` represents different depths, we can plot the results as follows:


.. code::

    from mtuq.graphics import plot_misfit_depth
    plot_misfit_depth(filename, ds, origins)


.. image:: images/20090407201255351_misfit_depth.png
  :width: 200 



Tradeoffs with depth
""""""""""""""""""""

To display tradeoffs with depth, we can add the following keyword arguments:

.. code::

    plot_misfit_depth(filename, ds, origins, show_tradeoffs=True, show_magnitudes=True)


.. image:: images/20090407201255351_misfit_depth_tradeoffs.png
  :width: 200 



Plotting misfit versus hypocenter
"""""""""""""""""""""""""""""""""

.. warning::

   1D Green's functions are used for the hypocenter grid search example. 

   (3D Green's functions would provide more meaningful results, but are too large for remote hosting.)



If `origins` represents different hypocenters, we can plot the results as follows:

.. code::

    from mtuq.graphics import plot_misfit_latlon
    plot_misfit_latlon(filename, ds, origins)


.. image:: images/20090407201255351_misfit_latlon.png
  :width: 200 



Tradeoffs with hypocenter
"""""""""""""""""""""""""

.. warning::

   1D Green's functions are used for the hypocenter grid search example. 

   (3D Green's functions would provide more meaningful results, but are too large for remote hosting.)



To see how orientation trade offs with hypocenter, we can add the following keyword argument:

.. code::

    plot_misfit_latlon(filename, ds, origins, show_tradeoffs=True)


.. image:: images/20090407201255351_misfit_latlon_tradeoffs.png
  :width: 200 



Source code
"""""""""""

The above figures can be reproduced by running `GridSearch.DoubleCouple+Magnitude+Depth.py <https://github.com/mtuqorg/mtuq/blob/master/examples/GridSearch.DoubleCouple+Magnitude+Depth.py>`_ and `GridSearch.DoubleCouple+Magnitude+Hypocenter.py <https://github.com/mtuqorg/mtuq/blob/master/examples/GridSearch.DoubleCouple+Magnitude+Hypocenter.py>`_. 

