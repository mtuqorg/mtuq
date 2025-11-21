
import numpy as np

from copy import deepcopy
from mtuq.misfit.waveform import level0, level1, level2
from mtuq.misfit.waveform._stats import estimate_sigma, calculate_norm_data, _flatten
from mtuq.util import Null, iterable, warn
from mtuq.util.math import isclose, list_intersect, list_intersect_with_indices
from mtuq.util.signal import check_padding, get_components, isempty


class WaveformMisfit(object):
    """ Waveform misfit function

    Evaluates misfit between data and synthetics using time shifts followed by
    waveform differences. This approach, due to `Zhao1994` and `Zhu1996`, has 
    become widely used in regional seismology.  See `docs/references` for more
    information.

    .. rubric:: Usage

    Evaluating misfit is a two-step procedure. 

    First, the user supplies parameters such as the type of norm (see below for
    detailed argument description):

    .. code::

        misfit = Misfit(**parameters)

    Second, the user supplies data, Green's functions, and sources:

    .. code::

        values = misfit(data, greens, sources)

    During misfit evaluation, synthetics are then generated and compared with 
    data, and a NumPy array of  misfit values is returned of the same length as
    `sources`.


    .. rubric:: Parameters

    ``norm`` (`str`)

    - ``'L2'``: conventional L2 norm (fast)
    ..  r1**2 + r1**2 + ...

    - ``'L1'``: conventional L1 norm (slow)
    ..  \\|r1\\| + \\|r2\\| + ...

    - ``'hybrid'``: hybrid L1-L2 norm (much faster than L1 but still robust)
    ..  (r11**2 + r12**2 + ...)**0.5 + (r21**2 + r22**2 + ...)**0.5 + ...


    ``level`` (`int`): optimization level 
    (see further details below)


    ``time_shift_groups`` (`list`)

    - ``['ZRT']``: forces all three components to have the same time shift

    - ``['ZR','T'``]: forces vertical and radial components to have the same
      time shift, while transverse time shift is allowed to vary independently

    - ``['Z','R','T']``: allows time shifts of all three components to vary
      independently

    ``time_shift_min`` (`float`): minimum allowable time shift (s)

    ``time_shift_max`` (`float`): maximum allowable time shift (s)


    .. note:: 

      *Convention* : A positive time shift means synthetics are arriving too 
      early and need to be shifted forward in time to match the observed data.


    .. rubric:: Optimization Levels

    Because waveform misfit evaluation is the most computationally expensive 
    task, we have implemented three different versions: 

    - a readable pure Python version (``mtuq.misfit.level0``)

    - a fast pure Python version (``mtuq.misfit.level1``)

    - a very fast numba version (``mtuq.misfit.level2``)

    - a very fast Cython version (``mtuq.misfit.level3``)


    While having exactly the same input argument syntax, the following versions
    differ in the following ways:

    - ``level=0`` provides a reference for understanding what the code is doing
      and for checking the correctness of the fast implementations

    - ``level=1`` is an optimized pure Python implementation which provides 
      significant computational savings for `len(sources)` > 100. This
      version is the closest to `Zhu1996`'s original C software.

    - ``level=2`` is an optimized numba implementation, in which a Python 
      wrapper is used to combine ObsPy traces into multidimensional arrays.
      These arrays are passed to a numba routine, which does the
      main computational work. Unlike the previous two versions, this 
      implementation requires that all ObsPy traces have the same time
      discretization.

    - ``level=3`` is an optimized Cython implementation, in which a Python 
      wrapper is used to combine ObsPy traces into multidimensional arrays.
      These arrays are passed to a Cython routine, which does the
      main computational work. This implementation also requires that all 
      ObsPy traces have the same time discretization.


    .. note:: 

      Cython extension modules are no longer automatically compiled during
      installation, but can be manually compiled via `build_ext.sh`.

    """

    def __init__(self,
        norm='hybrid',
        time_shift_groups=['ZRT'],
        time_shift_min=0.,
        time_shift_max=0.,
        level=2,
        normalize=True,
        verbose=2,
        ):
        """ Function handle constructor
        """

        if norm.lower()=='hybrid':
            norm = 'hybrid'

        assert norm in ['L1', 'L2', 'hybrid'],\
            ValueError("Bad input argument: norm")

        assert time_shift_min <= 0.,\
            ValueError("Bad input argument: time_shift_min")

        assert time_shift_max >= 0.,\
            ValueError("Bad input argument: time_shift_max")

        if norm=='L1':
            warn(
                "Consider using norm='hybrid', which is much faster than "
                "norm='L1' but still robust against outliers."
                )

        if type(time_shift_groups) not in (list, tuple):
            raise TypeError

        for group in time_shift_groups:
            for component in group:
                assert component in ['Z','R','T'],\
                    ValueError("Bad input argument")

        assert level in [0,1,2]

        self.level = level
        self.norm = norm

        self.time_shift_min = time_shift_min
        self.time_shift_max = time_shift_max
        self.time_shift_groups = time_shift_groups

        self.normalize = normalize

        self.verbose = verbose


    def __call__(self, data, greens, sources, progress_handle=Null(), 
        normalize=None, set_attributes=False, level=None):
        """ Evaluates misfit on given data
        """
        if normalize is None:
            normalize = self.normalize

        if level is None:
            level = self.level

        assert level in [0,1,2,3]

        # normally misfit is evaluated over a grid of sources; `iterable`
        # makes things work if just a single source is given
        sources = iterable(sources)

        # checks that dataset is nonempty
        if isempty(data):
            warn("Empty data set. No misfit evaluations will be carried out")
            return np.zeros((len(sources), 1))

        # checks that the container lengths are consistent
        if len(data) != len(greens):
            raise Exception("Inconsistent container lengths\n\n  "+
                "len(data): %d\n  len(greens): %d\n" %
                (len(data), len(greens)))
 

        # checks that optional Green's function padding is consistent with time 
        # shift bounds
        check_padding(greens, self.time_shift_min, self.time_shift_max)

 
        if level==0 or set_attributes:
            return level0.misfit(
                data, greens, sources, self.norm, self.time_shift_groups, 
                self.time_shift_min, self.time_shift_max, progress_handle,
                normalize=normalize, set_attributes=set_attributes)

        if level==1:
            return level1.misfit(
                data, greens, sources, self.norm, self.time_shift_groups, 
                self.time_shift_min, self.time_shift_max, progress_handle,
                normalize=normalize)

        if level==2:
            return level2.misfit(
                data, greens, sources, self.norm, self.time_shift_groups,
                self.time_shift_min, self.time_shift_max, progress_handle,
                normalize=normalize)

        if level==3:
            return level2.misfit(
                data, greens, sources, self.norm, self.time_shift_groups,
                self.time_shift_min, self.time_shift_max, progress_handle,
                normalize=normalize, ext='Cython') 


    def collect_attributes(self, data, greens, source, normalize=False):
        """ Collects misfit, time shifts and other attributes corresponding to 
        each trace
        """
        # checks that dataset is nonempty
        if isempty(data):
            warn("Empty data set. No attributes will be returned")
            return []

        # checks that optional Green's function padding is consistent with time 
        # shift bounds
        check_padding(greens, self.time_shift_min, self.time_shift_max)

        synthetics = greens.get_synthetics(
            source, components=data.get_components(), stats=data.get_stats(),
            mode='map', inplace=True)

        # attaches attributes to synthetics
        _ = level0.misfit(
            data, greens, iterable(source), self.norm, self.time_shift_groups,
            self.time_shift_min, self.time_shift_max, msg_handle=Null(),
            normalize=normalize, set_attributes=True)

        # collects attributes
        attrs = []
        for stream in synthetics:
            attrs += [{}]
            for trace in stream:
                component = trace.stats.channel[-1]
                if component in attrs[-1]:
                    print('Warning multiple traces for same component')
                if hasattr(trace, 'attrs'):
                    attrs[-1][component] = trace.attrs

        return deepcopy(attrs)


    def collect_synthetics(self, data, greens, source, normalize=False, mode=2):
        """ Collects synthetics with misfit, time shifts and other attributes attached
        """

        if mode==1: 
            # returns synthetics from all stations and components
            components = [['Z','R','T']]*len(data)
 
        elif mode==2:
            # returns synthetics only from stations and components which exist in
            # observed data
            components = data.get_components()
 
        elif mode==3:
            # returns synthetics only from stations and components which exist
            # and are included in misfit function evaluation
            _active = []
            for group in self.time_shift_groups:
                for component in group:
                    _active.append(component)
 
            components = []
            for _exist in data.get_components():
                components.append(list_intersect(_active, _exist))
 

        # checks that dataset is nonempty
        if isempty(data):
            warn("Empty data set. No attributes will be returned")
            return []

        # checks that optional Green's function padding is consistent with time 
        # shift bounds
        check_padding(greens, self.time_shift_min, self.time_shift_max)


        synthetics = greens.get_synthetics(
            source, components=components, stats=data.get_stats(),
            mode='map', inplace=True)

        # attaches attributes to synthetics
        _ = level0.misfit(
            data, greens, iterable(source), self.norm, self.time_shift_groups,
            self.time_shift_min, self.time_shift_max, msg_handle=Null(),
            normalize=False, set_attributes=True)

        return deepcopy(synthetics)


    def _description(self):
        _string = ''

        if self.verbose > 1:
            if self.norm=='L1':
                formula = 'Σ ∫ |d(t) - s(t-t_s)| dt'

            elif self.norm=='L2':
                formula = 'Σ ∫ |d(t) - s(t-t_s)|² dt'

            elif self.norm=='hybrid':
                formula = 'Σ √(∫ |d(t) - s(t-t_s)|² dt)'

            if self.normalize:
                formula = '('+formula+')' + ' / NF'

                if self.norm=='L1':
                    NF = 'Σ ∫ |d(t)| dt'

                elif self.norm=='L2':
                    NF = 'Σ ∫ |d(t)|² dt'

                elif self.norm=='hybrid':
                    NF = 'Σ √(∫ |d(t)|² dt)'

            _string += \
f"""\
    Misfit function evaluates
      {formula}

    where the sum is over components
      {', '.join(_flatten(self.time_shift_groups))}

    and where
      d(t) is observed data
      s(t) is synthetic data
"""

            if self.time_shift_min != self.time_shift_max:
                _string +=\
                    f'      t_s is a cross-correlation time shift\n'

            if self.normalize:
                _string +=\
                    f'      NF = {NF} is a normalization factor\n'

            _string += '\n'


        _type = type(self).__name__
        _level = {
            0: 'readable pure Python',
            1: 'fast pure Python',
            2: 'numba',
            3: 'Cython [deprecated]',
            }[self.level]

        _string += '\n'.join([
            f'    Misfit function type:\n    {_type}\n',
            f'    Misfit function implementation:\n    {_level}\n',
            ])

        return _string

         

#
# for backward compatibility
#
Misfit = WaveformMisfit


