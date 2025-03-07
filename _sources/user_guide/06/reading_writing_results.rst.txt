
Reading and writing grid search results
=======================================

A grid search 

.. code::

    ds = grid_search(data, greens, misfit, origins, sources)

returns a data structure `ds` that contains both misfit values and grid points.

To write this data structure to disk:

.. code::

    ds.save(filename)

Depending on whether the grid points are regularly- or irregularly-spaced, a NetCDF or HDF5 file will be written. Subsequently, this file can be read back in using `open_ds`, which tries to automatically determine file format:

.. code::

    from mtuq import open_ds
    ds = open_ds(filename)

Alternatively, users can specify file type through the `format` keyword argument, as in the examples below.



Details on regularly-spaced grids
---------------------------------

A search over reguarly-spaced `sources`

.. code::

    da = grid_search(data, greens, misfit, origins, sources)

returns an xarray DataArray `da`.

To write the grid search output `da` to disk as a NetCDF file:

.. code::

    da.save('output.nc')


To read the NetCDF file back from disk:

.. code::

    from mtuq import open_ds
    da = open_ds('output.nc', format='NetCDF')


Before or after writing to disk, `da` can be passed to visualization utilities or manipulated using `xarray methods <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_.



Details on irregularly-spaced grids
-----------------------------------

A search over irregularly-spaced `sources`

.. code::

    df = grid_search(data, greens, misfit, origins, sources)

returns a pandas DataFrame `df`.

To write the grid search output `df` to disk as an HDF5 file:

.. code::

    df.save('output.hf5')


To read the HDF5 file back from disk:

.. code::

    from mtuq import open_ds
    df = open_ds('output.hf5', format='HDF5')


Before or after writing to disk, `df` can be passed to visualization utilities or manipulated using `pandas methods <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.


