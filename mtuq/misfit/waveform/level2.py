"""
Waveform misfit module (fast Python/C version)

See ``mtuq/misfit/waveform/__init__.py`` for more information
"""

import numpy as np
import time
import warnings

from copy import deepcopy
from mtuq.misfit.waveform._stats import _flatten, calculate_norm_data
from mtuq.misfit.waveform.level1 import correlate
from mtuq.util.math import to_mij, to_rtp
from mtuq.util.signal import get_components, get_time_sampling
from mtuq.misfit.waveform import ext_numba


def misfit(data, greens, sources, norm, time_shift_groups,
    time_shift_min, time_shift_max, msg_handle, debug_level=0,
    normalize=False, ext='numba'):
    """
    Data misfit function (fast Python/C version)

    See ``mtuq/misfit/waveform/__init__.py`` for more information
    """

    if normalize:
        components = _flatten(time_shift_groups)
        norm_data = calculate_norm_data(data, norm, components)

    #
    # collect metadata
    #
    nt, dt = _get_time_sampling(data)
    stations = _get_stations(data)
    components = _get_components(data)


    # collect user-supplied data weights
    weights = _get_weights(data, stations, components)

    # which components will be used to determine time shifts? (boolean array)
    groups = _get_groups(time_shift_groups, components)


    #
    # collapse main structures into NumPy arrays
    #
    data = _get_data(data, stations, components)
    greens = _get_greens(greens, stations, components)
    sources = _to_array(sources)

    # sanity checks
    _check(data, greens, sources)


    #
    # cross-correlate data and synthetics
    #
    padding = _get_padding(time_shift_min, time_shift_max, dt)
    data_data = _autocorr_1(data)
    greens_greens = _autocorr_2(greens, padding)
    greens_data = _corr_1_2(data, greens, padding)

    if norm=='hybrid':
        hybrid_norm = 1
    else:
        hybrid_norm = 0

    #
    # collect message attributes
    #
    try:
        msg_args = [getattr(msg_handle, attrib) for attrib in 
            ['start', 'stop', 'percent']]
    except:
        msg_args = [0, 0, 0]

    #
    # call extension
    #

    start_time = time.time()

    if norm in ['L2', 'hybrid'] and ext.lower()=='numba':
        handle = ext_numba.misfit_L2

    elif norm in ['L1'] and ext.lower()=='numba':
        handle = ext_numba.misfit_L2

    elif norm in ['L2', 'hybrid'] and ext.lower()=='cython':
        from mtuq.misfit.waveform import ext_cython
        handle = ext_cython.misfit_L2

    else:
        raise NotImplementedError

    results = handle(
       data_data, greens_data, greens_greens, sources, groups, weights,
       hybrid_norm, dt, padding[0], padding[1], debug_level, *msg_args)

    if debug_level > 0:
      print('\n  Elapsed time (extension module) (s): %f' % \
          (time.time() - start_time))

    if normalize:
        results /= norm_data

    return results


#
# utility functions
#

def _get_time_sampling(dataset):
    for stream in dataset:
        if len(stream) > 0:
            return get_time_sampling(stream)


def _get_padding(time_shift_min, time_shift_max, dt):
    padding_left = int(round(+time_shift_max/dt))
    padding_right = int(round(-time_shift_min/dt))
    return [padding_left, padding_right]


def _get_greens(greens, stations, components):
    Ncomponents = len(components)
    Nstations = len(stations)
    Npts = len(greens[0][0])

    Ngreens = 0
    if greens[0].include_mt:
        Ngreens += 6
    if greens[0].include_force:
        Ngreens += 3

    array = np.zeros((
        Nstations,
        Ncomponents,
        Ngreens,
        Npts,
        ))

    for _i, station in enumerate(stations):
        tensor = greens.select(station)[0]

        # fill in array
        tensor._set_components(components)
        array[_i, :, :, :] = tensor._array

    return array


def _get_data(data, stations, components):
    # Collects numeric trace data from all streams as a single NumPy array;
    # compared with iterating over streams and traces, provides a potentially
    # faster way of accessing numeric trace data

    #.. warning::

    #    Requires that all streams have the same time discretization
    #    (or else an error is raised)

    nt, dt = _get_time_sampling(data)

    ns = len(stations)
    nc = len(components)
    array = np.zeros((ns, nc, nt))

    for _i, station in enumerate(stations):
        stream = data.select(station)[0]
        for _j, component in enumerate(components):
            try:
                trace = stream.select(component=component)[0]
            except:
                continue
            array[_i, _j, :] = trace.data

    return array



def _get_components(data):
    components = list()
    for stream in data:
        components.extend(get_components(stream))
    components = list(set(components))

    components_sorted = []
    if 'Z' in components:
        components_sorted += ['Z']
    if 'R' in components:
        components_sorted += ['R']
    if 'T' in components:
        components_sorted += ['T']

    return components_sorted


def _get_weights(data, stations, components):
    # user-supplied data weights

    Ncomponents = len(components)
    Nstations = len(stations)

    weights = np.ones((
        Nstations,
        Ncomponents,
        ))

    for _i, station in enumerate(stations):
        for _j, component in enumerate(components):

            stream = data.select(station)[0]
            stream = stream.select(component=component)

            if len(stream) > 0:
                try:
                    weights[_i, _j] = stream[0].attrs.weight
                except AttributeError as e:
                    #warnings.warn('Weight not attached', UserWarning)
                    weights[_i, _j] = 1.
            else:
                weights[_i, _j] = 0.

    return weights


def _get_mask(data, stations, components):
    # which components are absent from the data? (boolean array)

    Ncomponents = len(components)
    Nstations = len(stations)

    mask = np.ones((
        Nstations,
        Ncomponents, 
        ))

    for _i, station in enumerate(stations):
        for _j, component in enumerate(components):

            stream = data.select(station)[0]
            stream = stream.select(component=component)

            if len(stream)==0:
                mask[_i, _j] = 0.

    return mask


def _get_stations(data):
    stations = []
    for stream in data:
        if len(stream)==0:
            continue
        stations += [stream.station]
    return stations


def _get_groups(groups, components):
    Ncomponents = len(components)
    Ngroups = len(groups)

    array = np.zeros((
        Ngroups, 
        Ncomponents, 
        ))

    for _i, group in enumerate(groups):
        for _j, component in enumerate(components):
            if component in group:
                array[_i, _j] = 1

    return array


def _check(data, greens, sources):
    # array shape sanity checks

    if data.shape[0] != greens.shape[0]:
        print()
        print('Number of stations (data): %d' % data.shape[0])
        print('Number of stations (Green''s): %d' % data.shape[0])
        print()
        raise TypeError('Inconsistent shape')

    if data.shape[1] != greens.shape[1]:
        print()
        print('Number of components (data): %d' % data.shape[1])
        print('Number of components (Green''s): %d' % data.shape[1])
        print()
        raise TypeError('Inconsistent shape')

    if greens.shape[2] != sources.shape[1]:
        print()
        print('Number of Green''s functions in linear combination: %d' % greens.shape[2])
        print('Number of weights in linear combination: %d' % sources.shape[1])
        print()


def _to_array(sources):
    dims = sources.dims
    df = sources.to_dataframe()

    if _type(dims)=='MomentTensor':
        return np.ascontiguousarray(to_mij(
            df['rho'].to_numpy(),
            df['v'].to_numpy(),
            df['w'].to_numpy(),
            df['kappa'].to_numpy(),
            df['sigma'].to_numpy(),
            df['h'].to_numpy(),
            ))

    elif _type(dims)=='Force':
        return np.ascontiguousarray(to_rtp(
            df['F0'].to_numpy(),
            df['phi'].to_numpy(),
            df['h'].to_numpy(),
            ))


def _type(dims):
    if 'rho' in dims\
       and 'v' in dims\
       and 'w' in dims\
       and 'kappa' in dims\
       and 'sigma' in dims\
       and 'h' in dims:
        return 'MomentTensor'

    elif 'F0' in dims\
       and 'phi' in dims\
       and 'h' in dims:
        return 'Force'

    else:
        raise ValueError


#
# cross-correlation utilities
#

def _corr_1_2(data, greens, padding):
    # correlates 1D and 2D data structures
    Ncomponents = greens.shape[1]
    Nstations = greens.shape[0]
    Ngreens = greens.shape[2]

    corr = np.zeros((
        Nstations,
        Ncomponents,
        Ngreens,
        padding[0]+padding[1]+1,
        ))

    for _i in range(Nstations):
        for _j in range(Ncomponents):
            for _k in range(Ngreens):

                corr[_i, _j, _k, :] = correlate(
                    greens[_i, _j, _k, :], data[_i, _j, :])

    return corr


def _autocorr_1(data):
    # autocorrelates 1D data strucutres (reduces to dot product)
    Ncomponents = data.shape[1]
    Nstations = data.shape[0]

    corr = np.zeros((
        Nstations,
        Ncomponents,
        ))

    for _i in range(Nstations):
        for _j in range(Ncomponents):
            corr[_i, _j] = np.dot(
                data[_i, _j, :], data[_i, _j, :])

    return corr


def _autocorr_2(greens, padding):
    # autocorrelates 2D data structures

    Ncomponents = greens.shape[1]
    Nstations = greens.shape[0]
    Ngreens = greens.shape[2]
    Npts = greens.shape[3]

    ones = np.pad(np.ones(Npts), 
        (padding[0], padding[1]), 'constant')

    corr = np.zeros((
        Nstations,
        Ncomponents, 
        padding[0]+padding[1]+1, 
        Ngreens, 
        Ngreens,
        ))

    for _i in range(Nstations):
        for _j in range(Ncomponents):
            for _k1 in range(Ngreens):
                for _k2 in range(Ngreens):

                    if _k1<=_k2:
                        corr[_i, _j, :, _k1, _k2] = correlate(
                            greens[_i, _j, _k1, :]*greens[_i, _j, _k2, :],
                            ones)

                    else:
                        corr[_i, _j, :, _k1, _k2] = corr[_i, _j, :, _k2, _k1]

    return corr


