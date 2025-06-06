
import csv
import obspy
import numpy as np
import warnings

from os import listdir
from copy import deepcopy
from io import TextIOBase
from obspy import taup
from obspy.geodetics import gps2dist_azimuth
from os.path import basename, exists, isdir, join
from mtuq.util import AttribDict, warn
from mtuq.util.cap import WeightParser, taper
from mtuq.util.signal import cut, get_arrival, m_to_deg, _window_warnings


class ProcessData(object):
    """ An attempt at a one-size-fits-all data processing class


    .. rubric :: Usage

    Processing data is a two-step procedure. First, the user supplies parameters
    (see available choices below) to create a data processing function:

    .. code::

        function = ProcessData(**parameters)

    Second, an ObsPy stream is given as input to the data processing function
    and a processed stream returned as output:

    .. code::

        processed_stream = function(stream)


    Data processing can also be applied to an entire ``Dataset`` at once:

    .. code::

        processed_dataset = dataset.map(function)


    See `mtuq/examples/` for further illustration.


    .. rubric :: Parameters

    ``filter_type`` (`str`)

    - ``'bandpass'``
      Butterworth-Bandpass (uses `obspy.signal.filter.bandpass`)

    - ``'lowpass'``
      Butterworth-Lowpass (uses `obspy.signal.filter.lowpass`)

    - ``'highpass'``
      Butterworth-Highpass (uses `obspy.signal.filter.highpass`)

    - ``None``
      no filter will be applied


    ``pick_type`` (`str`)

    - ``'taup'``
      calculates P, S travel times from Tau-P model
      (uses `obspy.taup.TauPyModel.get_arrival_times`)

    - ``'FK_metadata'``
      reads P, S travel times from FK metadata

    - ``'CPS_metadata'``
      reads P, S travel times from CPS metadata

    - ``'SAC_metadata'``
      reads P, S travel times from SAC metadata fields `t5`, `t6`

    - ``'user_supplied'``
      reads P, S travel times from columns 8, 10 of `capuaf_file`

    - ``None``
      no P,S travel times will be calculated


    ``window_type`` (`str`)

    - ``'body_wave'``
      regional-distance body-wave window

    - ``'surface_wave'``
      regional-distance surface-wave window

    - ``'group_velocity'``
      surface-wave window centered on given group velocity

    - ``None``
      no windows will be applied


    ``apply_statics`` (`bool`)
    Whether or not to apply static time shifts from columns 11-12 of `capuaf_file`


    ``apply_weights`` (`bool`)
    Whether or not to apply objective function weights from columns 3-8 of `capuaf_file`


    ``apply_scaling`` (`bool`)
    Whether or not to apply distance-dependent amplitude scaling


    ``apply_padding`` (`bool`)
    Whether or not to use longer Green's window relative to observed data 
    window (allows for more accurate cross-correlations)


    .. rubric:: Other input arguments that may be required, depending on the above

    ``freq_min`` (`float`)
    Required for `filter_type=bandpass`

    ``freq_max`` (`float`)
    Required for `filter_type=bandpass`

    ``freq`` (`float`)
    Required for `filter_type=lowpass` or `filter_type=highpass`

    ``window_length`` (`float`)
    Window length in seconds

    ``group_velocity`` (`float`)
    Group velocity in m/s, required for `window_type=group_velocity`

    ``window_alignment`` (`float`)
    Optional window alignment for `window_type=group_velocity`
    (`float` between 0. and 1.)

    ``time_shift_min`` (`float`)
    Required for `apply_padding=True`

    ``time_shift_max`` (`float`)
    Required for `apply_padding=True`

    ``taup_model`` (`str`)
    Name of built-in ObsPy TauP model or path to custom ObsPy TauP model,
    required for `pick_type=taup`

    ``FK_database`` (`str`)
    Path to FK database, required for `pick_type=FK_metadata`

    ``CPS_database`` (`str`)
    Path to CPS database, required for `pick_type=CPS_metadata`

    ``capuaf_file`` (`str`)
    Path to `CAPUAF`-style text file, required for `pick_type=user_supplied`

    """

    def __init__(self,

                 filter_type=None,
                 window_type=None,
                 pick_type=None,

                 # for filter_type='Bandpass'
                 # (specify corners in terms of frequency or period, but not both)
                 freq_min=None,
                 freq_max=None,
                 period_min=None,
                 period_max=None,

                 # for filter_type='Lowpass' or filter_type='Highpass'
                 # (specify corner in terms of frequency or period, but not both)
                 freq=None,
                 period=None,

                 # window parameters
                 window_length=None,
                 apply_padding=False,
                 apply_statics=False,
                 time_shift_min=None,
                 time_shift_max=None,

                 # data weighting parameters
                 # (control contribution of particular traces or components to the
                 # data misfit function)
                 apply_weights=True,
                 apply_scaling=True,
                 scaling_power=None,
                 scaling_coefficient=None,
                 capuaf_file=None,

                 # P and S pick parameters
                 # (default body_wave and surface_wave windows are defined relative to
                 # P and S picks, which can be calculated on the fly from tau-P or
                 # read from FK or CPS database)
                 taup_model=None,
                 FK_database=None,
                 FK_model=None,
                 CPS_database=None,
                 CPS_model=None,

                 # any user-supplied keyword arguments not included above go into
                 # this dictionary (can be helpful for user customization)
                 **parameters):

        if not filter_type:
            warn("No filter will be applied")

        if not window_type:
            warn("No windows will be applied")

        if filter_type:
            filter_type = filter_type.lower()
        self.filter_type = filter_type

        if window_type:
            window_type = window_type.lower()
        self.window_type = window_type

        # note that we make pick_type case sensitive
        # (could be helpful because p,P s,S are meaningful differences?)
        self.pick_type = pick_type

        #
        # check filter parameters
        #

        if not self.filter_type:
            # nothing to check
            pass

        elif self.filter_type == 'bandpass':

            # filter corners can be specified in terms of either period [s]
            # or frequency [Hz], but not both
            if period_min is not None and\
               period_max is not None:

                assert freq_min is None
                assert freq_max is None
                freq_min = period_max**-1
                freq_max = period_min**-1

            else:
                assert freq_min is not None
                assert freq_max is not None

            assert 0 < freq_min
            assert freq_min < freq_max
            assert freq_max < np.inf

            self.freq_min = freq_min
            self.freq_max = freq_max

        elif self.filter_type == 'lowpass' or\
                self.filter_type == 'highpass':

            # filter corner can be specified in terms of either period [s]
            # or frequency [Hz], but not both
            if period is not None:
                assert freq is None
                freq = period*-1
            else:
                assert freq is not None

            assert 0 < freq < np.inf

            self.freq = freq

        else:
            raise ValueError('Bad parameter: filter_type')

        #
        # check window parameters
        #

        if not self.window_type:
            # nothing to check now
            pass

        elif self.window_type == 'body_wave':
            # regional body-wave window in the manner of Zhu1996
            assert pick_type is not None, "Must be defined: pick_type"
            assert window_length > 0.
            self.window_length = window_length

        elif self.window_type == 'surface_wave':
            # regional surface-wave window in the manner of Zhu1996
            assert pick_type is not None, "Must be defined: pick_type"
            assert window_length > 0.
            self.window_length = window_length

        elif self.window_type == 'group_velocity':
            assert 'group_velocity' in parameters
            assert parameters['group_velocity'] >= 0.
            self.group_velocity = parameters['group_velocity']
            self.window_alignment = getattr(
                parameters, 'window_alignment', 0.5)
            assert 0. <= self.window_alignment <= 1.
            assert window_length > 0.
            self.window_length = window_length

        elif self.window_type == 'min_max':
            assert 'v_min' in parameters
            assert 'v_max' in parameters
            assert 0. <= parameters['v_min']
            assert parameters['v_min'] <= parameters['v_max']
            assert parameters['v_max'] <= np.inf
            self.v_min = parameters['v_min']
            self.v_max = parameters['v_max']

            assert window_length >= 0.
            self.window_length = window_length
            _window_warnings(window_type, window_length)

        elif self.window_type == 'synthetic':
            # window type for synthetic examples
            assert window_length > 0.
            self.window_length = window_length

        else:
            raise ValueError('Bad parameter: window_type')

        if apply_statics:
            assert self.window_type is not None

        if apply_padding:
            assert self.time_shift_min <= 0., \
                ValueError("Bad parameter: time_shift_min")

            assert self.time_shift_max >= 0., \
                ValueError("Bad parameter: time_shift_max")

            self.time_shift_min = time_shift_min
            self.time_shift_max = time_shift_max

        self.apply_padding = apply_padding
        self.apply_statics = apply_statics

        #
        # check phase pick parameters
        #

        if not self.pick_type:
            # nothing to check now
            pass

        elif self.pick_type == 'taup':
            assert taup_model is not None
            self.taup_model = taup_model
            self._taup = taup.TauPyModel(self.taup_model)

        elif self.pick_type == 'FK_metadata':
            assert FK_database is not None
            assert exists(FK_database)

            if FK_model is None:
                FK_model = basename(FK_database)

            self.FK_database = FK_database
            self.FK_model = FK_model

        elif self.pick_type == 'CPS_metadata':
            assert CPS_database is not None
            assert exists(CPS_database)
            print(CPS_database)
            if CPS_model is None:
                CPS_model = basename(CPS_database)
                # Remove model name if no model is provided
                # CPS_database = CPS_database[:-(len(CPS_model) + 1)]
                print(CPS_database)
            self.CPS_database = CPS_database
            self.CPS_model = CPS_model

        elif self.pick_type == 'SAC_metadata':
            pass

        elif self.pick_type == 'user_supplied':
            pass

        else:
            raise ValueError('Bad parameter: pick_type, %s' % self.pick_type)

        #
        # check weight parameters
        #
        self.apply_scaling = apply_scaling
        self.apply_weights = apply_weights

        if apply_scaling:
            if self.window_type == 'body_wave':
                if scaling_power is None:
                    scaling_power = 1.

                if scaling_coefficient is None:
                    scaling_coefficient = 1.e5

            else:
                if scaling_power is None:
                    scaling_power = 0.5

                if scaling_coefficient is None:
                    scaling_coefficient = 1.e5

            self.scaling_power = scaling_power
            self.scaling_coefficient = scaling_coefficient

        #
        # parse text files
        #
        if self.apply_statics or\
           self.apply_weights or\
           self.pick_type == 'user_supplied':

            assert capuaf_file is not None
            parser = WeightParser(capuaf_file)

        if self.apply_statics:
            self.statics = parser.parse_statics()

        if self.apply_weights:
            self.weights = parser.parse_weights()

        if self.pick_type == 'user_supplied':
            self.picks = parser.parse_picks()

    def __call__(self, traces, station=None, origin=None, overwrite=False):
        '''
        Carries out data processing operations on ObsPy traces or
        MTUQ GreensTensors
        '''
        if station is None:
            station = getattr(traces, 'station', None)

        if origin is None:
            origin = getattr(traces, 'origin', None)

        # overwrite existing data?
        if overwrite:
            traces = traces
        else:
            traces = deepcopy(traces)

        if not hasattr(traces, 'id'):
            raise Exception('Missing station identifier')
        id = traces.id

        # collect location information
        distance_in_m, azimuth, _ = gps2dist_azimuth(
            origin.latitude,
            origin.longitude,
            station.latitude,
            station.longitude)

        # collect time sampling information
        nt, dt = traces[0].stats.npts, traces[0].stats.delta

        # Tags can be added through dataset.add_tag to keep track of custom
        # metadata or support other customized uses. Here we use tags to
        # distinguish data from Green's functions and displacement time series
        # from velcoity time series
        if not hasattr(traces, 'tags'):
            raise Exception('Missing tags attribute')
        tags = traces.tags

        if 'units:m' in tags:
            # nothing to do
            pass

        elif 'units:cm' in tags:
            # convert to meters
            for trace in traces:
                trace.data *= 1.e-2
            index = tags.index('units:cm')
            tags[index] = 'units:m'

        else:
            warn('Units not specified.')

        for trace in traces:
            if not hasattr(trace, 'attrs'):
                trace.attrs = AttribDict()

        #
        # part 1: filter traces
        #

        if self.filter_type == 'bandpass':
            for trace in traces:
                trace.detrend('demean')
                trace.detrend('linear')
                trace.taper(0.05, type='hann')
                trace.filter('bandpass', zerophase=False,
                             freqmin=self.freq_min,
                             freqmax=self.freq_max)

        elif self.filter_type == 'lowpass':
            for trace in traces:
                trace.detrend('demean')
                trace.detrend('linear')
                trace.taper(0.05, type='hann')
                trace.filter('lowpass', zerophase=False,
                             freq=self.freq)

        elif self.filter_type == 'highpass':
            for trace in traces:
                trace.detrend('demean')
                trace.detrend('linear')
                trace.taper(0.05, type='hann')
                trace.filter('highpass', zerophase=False,
                             freq=self.freq)

        if 'type:velocity' in tags:
            # convert to displacement
            for trace in traces:
                trace.data = np.cumsum(trace.data)*dt
            index = tags.index('type:velocity')
            tags[index] = 'type:displacement'

        #
        # part 2a: apply distance scaling
        #

        if self.apply_scaling:
            for trace in traces:
                trace.data *=\
                    (distance_in_m/self.scaling_coefficient)**self.scaling_power

        #
        # part 2b: apply user-supplied data weights
        #
        if 'type:greens' in tags:
            pass

        elif self.apply_weights:
            for trace in traces:
                try:
                    component = trace.stats.channel[-1].upper()

                    if self.window_type == 'body_wave':
                        key = 'body_wave_'+component
                    else:
                        key = 'surface_wave_'+component

                    weight = self.weights[id][key]

                except:
                    weight = None

                if weight:
                    trace.attrs.weight = weight
                else:
                    traces.remove(trace)

        #
        # part 3: determine phase picks
        #

        if not self.pick_type:
            pass

        elif self.pick_type == 'user_supplied':
            picks = self.picks[id]

        else:
            picks = dict()

            if self.pick_type == 'taup':
                with warnings.catch_warnings():
                    # suppress obspy warning that gets raised even when taup is
                    # used correctly (someone should submit an ObsPy fix)
                    warnings.filterwarnings('ignore')
                    arrivals = self._taup.get_travel_times(
                        origin.depth_in_m/1000.,
                        m_to_deg(distance_in_m),
                        phase_list=['p', 's', 'P', 'S'])
                try:
                    picks['P'] = get_arrival(arrivals, 'p')
                except:
                    picks['P'] = get_arrival(arrivals, 'P')
                try:
                    picks['S'] = get_arrival(arrivals, 's')
                except:
                    picks['S'] = get_arrival(arrivals, 'S')

            elif self.pick_type == 'FK_metadata':
                sac_headers = obspy.read('%s/%s_%s/%s.grn.0' %
                                         (self.FK_database,
                                          self.FK_model,
                                          str(int(np.ceil(origin.depth_in_m/1000.))),
                                             str(int(np.ceil(distance_in_m/1000.)))),
                                         format='sac')[0].stats.sac
                picks['P'] = sac_headers.t1
                picks['S'] = sac_headers.t2

            elif self.pick_type == 'CPS_metadata':
                dep_desired = "{:06.1f}".format(
                    np.ceil(origin.depth_in_m/1000.) * 10)[:-2]

                # Review all folders in CPS Green's Function directory. Folder
                # names correspond with depth of source. Find the folder
                # with a value closest to the one we are after.
                all_entries = listdir(self.CPS_database)

                # Filter out folder names that are numeric
                numeric_folder_names = [entry for entry in all_entries
                                        if entry.isdigit() and isdir(join(self.CPS_database, entry))]

                # Convert numeric folder names to integers
                numeric_folder_names_int = [int(folder)
                                            for folder in numeric_folder_names]

                # Find depth closest to our desired value
                dep_folder = numeric_folder_names[numeric_folder_names_int.index(min(numeric_folder_names_int,
                                                                                     key=lambda x: abs(x - int(dep_desired))))]

                dst_desired = "{:07.1f}".format(
                    np.ceil(distance_in_m/1000.) * 10)[:-2]

                directory_path = self.CPS_database + '/' + dep_folder
                all_files = listdir(directory_path)
                filenames_without_extensions_inline = [
                    filename.split('.')[0] for filename in all_files]
                filenames_without_letters = [filename for filename in filenames_without_extensions_inline if not any(
                    char.isalpha() for char in filename)]
                filenames_unique = [entry[:5]
                                    for entry in list(set(filenames_without_letters))]
                filenames_unique_int = [int(filename)
                                        for filename in filenames_unique]
                dst_value = filenames_unique[filenames_unique_int.index(
                    min(filenames_unique_int, key=lambda x: abs(x - int(dst_desired))))]

                sac_headers = obspy.read('%s/%s/%s/%s%s.ZDD' %
                                         (self.CPS_database, self.CPS_model,
                                          dep_folder, dst_value, dep_folder),
                                         format='sac')[0].stats.sac

                picks['P'] = sac_headers.a
                picks['S'] = sac_headers.t0

            elif self.pick_type == 'SAC_metadata':
                sac_headers = traces[0].sac
                picks['P'] = sac_headers.t5
                picks['S'] = sac_headers.t6

        for trace in traces:

            #
            # part 4a: determine window start and end times
            #

            if self.window_type == 'body_wave':
                # regional body-wave window in the manner of Zhu1996
                # (closely based on CAP code)

                starttime = picks['P'] - 0.4*self.window_length
                endtime = starttime + self.window_length

                starttime += float(origin.time)
                endtime += float(origin.time)

            elif self.window_type == 'surface_wave':
                # regional surface-wave window in the manner of Zhu1996
                # (closely based on CAP code)

                starttime = picks['S'] - 0.3*self.window_length
                endtime = starttime + self.window_length

                starttime += float(origin.time)
                endtime += float(origin.time)

            elif self.window_type == 'group_velocity':
                # surface-wave window based on given group velocity [m/s]

                group_arrival = distance_in_m/self.group_velocity

                # alignment=0.0 - window starts at group arrival
                # alignment=0.5 - window centered on group arrival (default)
                # alignment=1.0 - window ends at group arrival
                # alignment = self.alignment

                starttime = group_arrival - self.window_length*self.window_alignment
                endtime = group_arrival + \
                    self.window_length*(1.-self.window_alignment)

                starttime += float(origin.time)
                endtime += float(origin.time)

            elif self.window_type == 'min_max':
                # experimental window type defined by minimum and maximum
                # group velocities [m/s]

                # WARNING - results in distance-dependent window lengths,
                # which may not work with other MTUQ functions

                starttime = distance_in_m/self.v_max
                endtime = distance_in_m/self.v_min

                # optionally, enforce minimum window length
                if endtime - starttime < self.window_length:
                    average_time = (starttime + endtime)/2.
                    starttime = average_time - self.window_length/2.
                    endtime = average_time + self.window_length/2.

                starttime += float(origin.time)
                endtime += float(origin.time)

            elif self.window_type == 'synthetic':
                # window type for synthetic examples
                # waveform starts at 0.0
                starttime = 0.0
                endtime = starttime + self.window_length
                starttime += float(origin.time)
                endtime += float(origin.time)

            else:
                starttime = trace.stats.starttime
                endtime = trace.stats.endtime

            #
            # part 4b: apply statics
            #

            # STATIC CONVENTION:  A positive static time shift means synthetics
            # are arriving too early and need to be shifted forward in time
            # (positive shift) to match the observed data

            if self.apply_statics:
                try:
                    # _component is a custom metadata attribute added by
                    # mtuq.io.clients

                    # Even though obspy.read doesn't return a stats.component
                    # attribute, "component" is still reserved by ObsPy,
                    # thus we use "_component" instead
                    component = trace.stats._component

                except:
                    # This way of getting the component from the channel is
                    # actually what is hardwired into ObsPy, and is implemented
                    # here as a fallback
                    component = trace.stats.channel[-1].upper()

                try:
                    if self.window_type == 'body_wave':
                        key = 'body_wave_'+component

                    else:
                        key = 'surface_wave_'+component

                    static = self.statics[id][key]
                    trace.attrs.static_shift = static

                except:
                    print('Error reading static time shift: %s' % id)
                    static = 0.

                if self.window_type is not None and\
                   'type:greens' in tags:

                    trace.stats.starttime += static

            #
            # part 4c: apply padding
            #

            # using a longer window for Green's functions than for data allows for
            # more accurate time-shift corrections

            if self.apply_padding and\
               'type:greens' in tags:

                starttime += self.time_shift_min
                endtime += self.time_shift_max

                trace.stats.npts_padding_left = int(
                    round(-self.time_shift_min/dt))
                trace.stats.npts_padding_right = int(
                    round(+self.time_shift_max/dt))

            #
            # part 4d: cut and taper trace
            #

            # cuts trace and adjusts metadata
            if self.window_type is not None:
                cut(trace, starttime, endtime)

            elif self.apply_statics and 'type:greens' in tags:
                print('Not implemented warning')

            taper(trace.data)

        return traces
