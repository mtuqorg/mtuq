

import numpy as np

from builtins import object
from pandas import DataFrame
from xarray import DataArray

from mtuq.util import asarray



class Grid(object):
    """
    A regularly-spaced grid defined by values along axes

    .. rubric:: Examples

    To cover the unit square with an `N`-by-`N` rectangular grid:

    .. code ::

       x = np.linspace(0., 1., N)
       y = np.linspace(0., 1., N)
       grid = Grid(dims=('x', 'y'), coords=(x, y))


    To parameterize the surface of the Earth with an `N`-by-`2N` Mercator grid:

    .. code ::          

       lat = np.linspace(-90., 90., N)
       lon = np.linspace(-180., 180., 2*N)
       grid = Grid(dims=('lat', 'lon'), coords=(lat, lon))


    .. rubric:: Iterating over grids

    Iterating over a grid is similar to iterating over a multidimensional 
    NumPy array.  The order of grid points is determined by the order of axes
    used to create the grid.  For instance, in the unit square example above, 
    ``'x'`` is the slow axis and ``'y'`` is the fast axis.

    If ``start`` and ``stop`` arguments are given when creating a grid,
    iteration will begin and end at these indices.  Otherwise, iteration will
    begin at the first index (`i=0`) and stop at the last index.


    .. rubric:: Accessing individual grid points

    Individual grid points can be accessed through the ``get`` and ``get_dict``
    methods.  

    ``get(i)`` returns the `i`-th grid point as a NumPy array.

    If a ``callback`` function was given when creating the grid, then ``get`` 
    returns the result of applying the callback to the `i`-th grid point.  This
    behavior can be overridden by supplying a callback function as a keyword
    argument to ``get`` itself.  If ``callback`` is ``None``, then no function 
    is applied.

    ``get_dict(i)`` returns the `i`-th grid point as a dictionary of coordinate
    axis names and coordinate values without applying any callback.

    """
    def __init__(self, dims=None, coords=None, start=0, stop=None, callback=None):
        # list of axis names
        self.dims = dims
        
        # corresponding list of axis coordinates
        self.coords = list(map(asarray, coords))

        # what is the length along each axis?
        shape = []
        for array in self.coords:
            shape += [len(array)]

        # what attributes would the grid have if stored as a numpy array?
        self.ndim = len(shape)
        self.shape = tuple(shape)

        # what part of the grid do we want to iterate over?
        self.start = start
        if stop:
            self.stop = stop
            self.size = stop-start
        else:
            self.stop = np.prod(shape)
            self.size = np.prod(shape)-start
        self.index = start

        self.callback = callback

 
    def to_array(self):
        """ Returns the entire set of grid points as a NumPy array
        """
        array = np.zeros((self.size, self.ndim))
        for _i in range(self.size):
            array[_i, :] = self.get(_i+self.start, callback=None)
        return array


    def to_dataarray(self, values=None):
        """ Returns the entire set of grid points as an `xarray.DataArray`
        """
        if values is None:
            values = np.empty(self.shape)
            values[:] = np.nan

        if values.size != self.size:
            raise Exception("Mismatch between values and grid shape")

        if values.shape != self.shape:
            values = np.reshape(values, self.shape)

        return DataArray(data=values, dims=self.dims, coords=self.coords)


    def to_dataframe(self, values=None):
        """ Returns the entire set of grid points as a `pandas.DataFrame`
        """
        if values is None:
            values = np.empty(self.size)
            values[:] = np.nan

        if values.size != self.size:
            raise Exception("Mismatch between values and grid shape")

        array = self.to_array()
        data_vars = {self.dims[_i]: array[:, _i]
            for _i in range(self.ndim)}
        data_vars.update({'values': values})

        return DataFrame(data_vars)


    def get(self, i, **kwargs):
        """ Returns `i`-th grid point

        .. rubric:: callback functions

        If a ``callback`` function was given when creating a grid, then 
        ``get`` returns the result of applying the callback to the 
        `i`-th grid point.  This behavior can be overridden by supplying a 
        callback function as a keyword argument to ``get`` itself.  
        If ``callback`` is ``None``, then no function is applied.
        """
        # optionally override default callback
        if 'callback' in kwargs:
            callback = kwargs['callback']
        else:
            callback = self.callback

        vals = self.coords
        array = np.zeros(self.ndim)

        for _k in range(self.ndim-1, -1, -1):
            val = vals[_k]
            array[_k] = val[int(i%len(val))]
            i/=len(val)

        if callback:
            return callback(*array)
        else:
            return array


    def get_dict(self, i):
        """ Returns `i`-th grid point grid as a dictionary of parameter names 
        and values
        """
        keys = self.dims
        vals = self.get(i, callback=None)

        return dict(zip(keys, vals))


    def partition(self, nproc):
        """ Partitions grid for parallel processing
        """
        if self.start!=0:
            raise Exception

        subsets = []
        for iproc in range(nproc):
            start = int(iproc*self.size/nproc)
            stop = int((iproc+1)*self.size/nproc)

            subsets += [Grid(
                self.dims, self.coords, start, stop, callback=self.callback)]
        return subsets


    def __len__(self):
        return self.size



    # the next two methods make it possible to iterate over the grid
    def __next__(self): 
        """ Advances iteration index
        """
        if self.index < self.stop:
           # get the i-th point in grid
           p = self.get(self.index)
        else:
            self.index = self.start
            raise StopIteration
        self.index += 1
        return p


    def __iter__(self):
        return self




class UnstructuredGrid(object):
    """ 
    An `UnstructuredGrid` is defined by lists of individual coordinate points, 
    which can be irregularly spaced

    .. rubric:: Example

    Unstructured grid consisting of `N` randomly-chosen points within the unit 
    square:

    .. code ::

       x = np.random.rand(N)
       y = np.random.rand(N)
       grid = UnstructuredGrid(dims=('x', 'y'), coords=(x, y))


    .. rubric:: Iterating over grids

    Iterating over an unstructured grid is similar to iterating over a list.

    If ``start`` and ``stop`` arguments are given when creating an unstructured
    grid, iteration will begin and end at these indices.  Otherwise, iteration
    will begin at the first index (`i=0`) and stop at the last index.


    .. rubric:: Accessing individual grid points

    Individual grid points can be accessed through the ``get`` and ``get_dict``
    methods.  

    ``get(i)`` returns the `i`-th grid point as a NumPy array.

    If a ``callback`` function is given when creating an unstructured grid, then
    ``get`` returns the result of applying the callback to the `i`-th grid point.
    This behavior can be overridden by supplying a callback function as a
    keyword argument to ``get`` itself.  If ``callback`` is ``None``, then no 
    function is applied.

    ``get_dict(i)`` returns the `i`-th grid point as a dictionary of coordinate
    axis names and coordinate values without applying any callback.


    """
    def __init__(self, dims=None, coords=None, start=0, stop=None, callback=None):
        # list of parameter names
        self.dims = dims

        # corresponding list of parameter values
        self.coords = list(map(asarray, coords))

        # there is no shape attribute because it is an unstructured grid,
        # however, ndim and size still make sense
        self.ndim = len(self.dims)
        size = len(self.coords[0])

        # check consistency
        for array in self.coords:
            assert len(array) == size

        # what part of the grid do we want to iterate over?
        self.start = start
        if stop:
            self.stop = stop
            self.size = stop-start
        else:
            self.stop = size
            self.size = size-start

        self.index = start

        # optional map from one parameterization to another
        self.callback = callback


    def to_array(self):
        """ Returns the entire set of grid points as a NumPy array
        """
        array = np.zeros((self.size, self.ndim))
        for _i in range(self.size):
            array[_i, :] = self.get(_i+self.start, callback=None)
        return array


    def to_dataframe(self, values=None):
        """ Returns the entire set of grid points as a `pandas.DataFrame`
        """
        if values is None:
            values = np.empty(self.size)
            values[:] = np.nan

        if values.size != self.size:
            raise Exception("Mismatch between values and grid shape")

        data_vars = {self.dims[_i]: self.coords[_i]
            for _i in range(self.ndim)}

        data_vars.update({'values': values})

        return DataFrame(data_vars)


    def get(self, i, **kwargs):
        """ Returns `i`-th grid point

        .. rubric:: callback functions

        If a ``callback`` function was given when creating a grid, then 
        ``get`` returns the result of applying the callback to the 
        `i`-th grid point.  This behavior can be overridden by supplying a 
        callback function as a keyword argument to ``get`` itself.  
        If ``callback`` is ``None``, then no function is applied.

        """
        # optionally override default callback
        if 'callback' in kwargs:
            callback = kwargs['callback']
        else:
            callback = self.callback

        i -= self.start
        vals = self.coords
        array = np.zeros(self.ndim)

        for _k in range(self.ndim):
            array[_k] = vals[_k][i]

        if callback:
            return callback(*array)
        else:
            return array


    def get_dict(self, i):
        """ Returns `i`-th grid point as a dictionary of parameter names and
        values
        """
        keys = self.dims
        vals = self.get(i, callback=None)

        return dict(zip(keys, vals))


    def partition(self, nproc):
        """ Partitions grid for parallel processing
        """
        subsets = []
        for iproc in range(nproc):
            start = int(iproc*self.size/nproc)
            stop = int((iproc+1)*self.size/nproc)

            coords = []
            for array in self.coords:
                coords += [array[start:stop]]
            subsets += [UnstructuredGrid(
                self.dims, coords, start, stop, callback=self.callback)]

        return subsets


    def __len__(self):
        return self.size


    # the next two methods make it possible to iterate over the grid
    def __next__(self): 
        """ Advances iteration index
        """
        if self.index < self.stop:
           # get the i-th point in grid
           p = self.get(self.index)
        else:
            self.index = self.start
            raise StopIteration
        self.index += 1
        return p


    def __iter__(self):
        return self
