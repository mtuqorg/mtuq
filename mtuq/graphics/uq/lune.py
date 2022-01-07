#
# For details about the eigenvalue lune, see
# Tape2012 - A geometric setting for moment tensors
# (https://doi.org/10.1111/j.1365-246X.2012.05491.x)
#

import numpy as np
import pandas
import xarray

from matplotlib import pyplot
from mtuq.grid_search import DataArray, DataFrame, MTUQDataArray, MTUQDataFrame
from mtuq.graphics.uq._gmt import _plot_lune_gmt
from mtuq.util import warn
from mtuq.util.math import lune_det, to_gamma, to_delta

from mtuq.graphics.uq.vw import\
    _misfit_vw_regular, _misfit_vw_random,\
    _likelihoods_vw_regular, _likelihoods_vw_random,\
    _marginals_vw_regular, _marginals_vw_random,\
    _variance_reduction_vw_regular, _magnitudes_vw_regular


def plot_misfit_lune(filename, ds, **kwargs):
    """ Plots misfit values on eigenvalue lune (requires GMT)

    .. rubric :: Required input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_lune.html>`_

    """ 
    _defaults(kwargs, {
        'colormap': 'viridis',
        })

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        misfit = _misfit_vw_regular(ds)

    elif issubclass(type(ds), DataFrame):
        misfit = _misfit_vw_random(ds)        

    _plot_lune(filename, misfit, **kwargs)



def plot_likelihood_lune(filename, ds, var, **kwargs):
    """ Plots maximum likelihood values on eigenvalue lune (requires GMT)

    .. rubric :: Required input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

    ``var`` (`float` or `array`):
    Data variance


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_lune.html>`_
    """
    _defaults(kwargs, {
        'colormap': 'hot_r',
        })

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        likelihoods = _likelihoods_vw_regular(ds, var)

    elif issubclass(type(ds), DataFrame):
        likelihoods = _likelihoods_vw_random(ds, var)

    _plot_lune(filename, likelihoods, **kwargs)



def plot_marginal_lune(filename, ds, var, **kwargs):
    """ Plots maximum likelihood values on eigenvalue lune (requires GMT)

    .. rubric :: Required input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

    ``var`` (`float` or `array`):
    Data variance


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_lune.html>`_
    """
    _defaults(kwargs, {
        'colormap': 'hot_r',
        })

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        marginals = _marginals_vw_regular(ds, var)

    elif issubclass(type(ds), DataFrame):
        marginals = _marginals_vw_random(ds, var)

    _plot_lune(filename, marginals, **kwargs)


def plot_variance_reduction_lune(filename, ds, data_norm, **kwargs):
    """ Plots variance reduction values on eigenvalue lune (requires GMT)

    .. rubric :: Required input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

    ``data_norm`` (`float`):
    Norm of data

    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_lune.html>`_

    """
    _defaults(kwargs, {
        'colormap': 'viridis_r',
        })

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        misfit = 100.*_variance_reduction_vw_regular(ds, data_norm)

    elif issubclass(type(ds), DataFrame):
        misfit = 100.*_variance_reduction_vw_random(ds, data_norm)

    _plot_lune(filename, misfit, **kwargs)


def plot_magnitude_tradeoffs_lune(filename, ds, **kwargs):
    """ Plots magnitude versus source type tradeoffs (requires GMT)

    .. rubric :: Required input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

    ``var`` (`float` or `array`):
    Data variance


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_lune.html>`_
    """
    _defaults(kwargs, {
        'colormap': 'gray',
        })

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        marginals = _magnitudes_vw_regular(ds)

    elif issubclass(type(ds), DataFrame):
        raise NotImplementedError

    _plot_lune(filename, marginals, **kwargs)


#
# backend
#

def _plot_lune(filename, da, show_best=True, show_tradeoffs=False,
    backend=_plot_lune_gmt, **kwargs):

    """ Plots DataArray values on the eigenvalue lune (requires GMT)

    .. rubric :: Keyword arguments

    ``colormap`` (`str`)
    Color palette used for plotting values 
    (choose from GMT or MTUQ built-ins)

    ``show_best`` (`bool`):
    Show where best-fitting moment tensor falls on lune

    ``show_tradeoffs`` (`bool`):
    Show how focal mechanism trades off with lune coordinates

    ``title`` (`str`)
    Optional figure title

    """
    if not issubclass(type(da), DataArray):
        raise Exception()


    best_vw = None
    lune_array = None

    if show_best:
        if 'best_vw' in da.attrs:
            best_vw = da.attrs['best_vw']
        else:
            warn("Best-fitting moment tensor not given")

    if show_tradeoffs:
        if 'lune_array' in da.attrs:
            lune_array = da.attrs['lune_array']
        else:
            warn("Focal mechanism tradeoffs not given")


    backend(filename, 
        to_gamma(da.coords['v']), 
        to_delta(da.coords['w']),
        da.values.transpose(), 
        best_vw=best_vw,
        lune_array=lune_array,
        **kwargs)


#
# utility functions
#

def _check(ds):
    """ Checks data structures
    """
    if type(ds) not in (DataArray, DataFrame, MTUQDataArray, MTUQDataFrame):
        raise TypeError("Unexpected grid format")


def _defaults(kwargs, defaults):
    for key in defaults:
        if key not in kwargs:
           kwargs[key] = defaults[key]


