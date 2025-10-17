
Grid search data structures
===========================

During a grid search

.. code::

    ds = grid_search(data, greens, misfit, origins, sources)


the misfit function is evaluated over all `origins` and `sources`.

The result is a data strucutre `ds` containing both misfit values and grid points.


Input data structures
---------------------

The `origins` data structure

- Is simply a list of `Origin` objects
- Allows for searching over depth, latitude, longitude and/or origin time, as specified by the attributes of the `Origin` objects in the list

The `sources` data structure

- Can be either a regularly-spaced `Grid` or an irregularly-spaced `UnstructuredGrid`
- Allows for searching over any subset of moment tensor space or force space, including these currently-implemented `moment tensor grids <https://mtuqorg.github.io/mtuq/user_guide/06/moment_tensor_and_force_grids.html#currently-implemented-moment-tensor-grids>`_ and `force grids <https://mtuqorg.github.io/mtuq/user_guide/06/moment_tensor_and_force_grids.html#currently-implemented-force-grids>`_


Output data structures
----------------------

If `sources` is regularly spaced, the output `ds` will be a subclass of `xarray DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_.

If `sources` is irregularly spaced, the output `ds` will be a subclass of `pandas DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.

See also `reading and writing grid search results <https://mtuqorg.github.io/mtuq/user_guide/06/reading_writing_results.html>`_.


