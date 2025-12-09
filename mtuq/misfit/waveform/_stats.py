
import numpy as np

from copy import deepcopy
from mtuq.util import warn
from mtuq.util.math import list_intersect_with_indices
from mtuq.util.signal import get_components


def calculate_norm_data(data, norm, components, apply_weights=True):
    """ Calculates the norm of the given waveform data
    """

    # Here we try to mimic the implementation in
    # mtuq/misfit/waveform/level0.py

    # Mimicking the misfit function implementation helps ensure consistency
    # between the misfit values (i.e. residual norms) and data norms

    # The idea here is to use the regular misfit function machinery, but set 
    # synthetics to zero, so that instead of the residual norm || d - s ||,
    # we get the data norm || d - 0 ||

    norm_data = 0.

    for _j, stream in enumerate(data):
        _components, indices = list_intersect_with_indices(
            get_components(stream), _flatten(components))

        if not indices:
            continue

        # time sampling scheme
        npts = stream[0].data.size
        dt = stream[0].stats.delta

        for _k in indices:
            trace = stream[_k]
            d = trace.data

            if norm=='L1':
                value = np.sum(np.abs(d))*dt

            elif norm=='L2':
                value = np.sum(d**2)*dt

            elif norm=='hybrid':
                value = np.sqrt(np.sum(d**2))*dt

            # optionally, applies user-supplied weights attached during
            # process_data()
            if apply_weights:
                try:
                    value *= trace.weight
                except Exception:
                    pass

            norm_data += value

    return norm_data


def estimate_sigma(data, greens, best_source, norm, components,
    time_shift_min, time_shift_max):
    """ Makes an a posteriori estimate of the data error standard deviation
    """

    # error checking
    assert norm in ('L1', 'L2')

    residuals = []
    for _j, d in enumerate(data):
        _components, indices = list_intersect_with_indices(
            get_components(d), components)

        if not indices:
            continue

        # generate synthetics
        greens[_j]._set_components(get_components(d))
        s = greens[_j].get_synthetics(best_source)

        # time sampling scheme
        npts = d[0].data.size
        dt = d[0].stats.delta

        padding_left = int(+time_shift_max/dt)
        padding_right = int(-time_shift_min/dt)
        npts_padding = padding_left + padding_right

        # array to hold cross correlations
        corr = np.zeros(npts_padding+1)

        #
        # calculate residuals
        # 

        corr[:] = 0.
        for _k in indices:
            corr += np.correlate(s[_k].data, d[_k].data, 'valid')

        npts_shift = padding_left - corr.argmax()
        time_shift = npts_shift*dt

        # what start and stop indices will correctly shift synthetics
        # relative to data?
        start = padding_left - npts_shift
        stop = start + npts

        for _k in indices:
            # subtract data from shifted synthetics
            r = s[_k].data[start:stop] - d[_k].data

            # sum the resulting residuals
            if norm=='L1':
                residuals += [np.sum(np.abs(r))*dt]

            elif norm=='L2':
                residuals += [np.sum(r**2)*dt]


    return np.mean(residuals)**0.5



def _flatten(lists):
    # TODO - better implementation?

    # example:
    #   converts ['ZR','T'] to ['Z','R','T']

    flattened = []
    for list in lists:
        for element in list:
            flattened.append(element)
    return flattened

