#
# all these functions reproduce functionality from
# github.com/uafseismo/capuaf
#


import csv
import numpy as np
import obspy

from collections import defaultdict
from copy import deepcopy
from io import TextIOBase
from mtuq.event import MomentTensor
from mtuq.util import AttribDict, warn
from mtuq.wavelet import EarthquakeTrapezoid
from os.path import exists


#
# the following functions allow reading in and performing other operations with
# CAP-style weight files. Such files can be used to control the weight on
# individual stations and components in a moment tensor inversion
#

class WeightParser(object):
    """ Parses CAPUAF-style text file

        For more information, see github.com/uafseismo/capuaf/doc/capuaf_manual
    """
    def __init__(self, file):
        if isinstance(file, TextIOBase):
            self._file = file
        else:
            assert exists(file)
            self._file = open(file, "r")


    def _parse_code(self, string):
        return '.'.join(string.split('.')[1:4])

    def _parse_polarity(self, string):
        if '/' in string:
            polarity = string.split('/')[-1]
        else:
            polarity = 0
        return polarity

    def parse_weights(self, remove_unused=True):

        self._file.seek(0)

        weights = defaultdict(AttribDict)

        reader = csv.reader(
            filter(lambda row: row[0]!='#', self._file),
            delimiter=' ',
            skipinitialspace=True)

        for row in reader:
            _code = self._parse_code(row[0])

            weights[_code]['body_wave_Z'] = float(row[2])
            weights[_code]['body_wave_R'] = float(row[3])
            weights[_code]['surface_wave_Z'] = float(row[4])
            weights[_code]['surface_wave_R'] = float(row[5])
            weights[_code]['surface_wave_T'] = float(row[6])

        if remove_unused:
            unused = []
            for _code, _weight in weights.items():
                if _weight['body_wave_Z']==_weight['body_wave_R']==\
                   _weight['surface_wave_Z']==_weight['surface_wave_R']==\
                   _weight['surface_wave_T']==0.:
                    unused += [_code]

            for _code in unused:
                weights.pop(_code)

        return weights


    def parse_picks(self):

        self._file.seek(0)

        picks = defaultdict(AttribDict)

        reader = csv.reader(
            filter(lambda row: row[0]!='#', self._file),
            delimiter=' ',
            skipinitialspace=True)

        for row in reader:
            _code = self._parse_code(row[0])
            picks[_code]['P'] = float(row[7])
            picks[_code]['S'] = float(row[9])

        return picks

    def parse_polarity(self, remove_unused=True):

        self._file.seek(0)

        polarities = []

        reader = csv.reader(
            filter(lambda row: row[0]!='#', self._file),
            delimiter=' ',
            skipinitialspace=True)

        for row in reader:
            if not float(row[2]) ==\
            float(row[3]) ==\
            float(row[4]) ==\
            float(row[5]) ==\
            float(row[6]) == 0:
                polarities+=[int(self._parse_polarity(row[0]))]

        return polarities


    def parse_statics(self):

        self._file.seek(0)

        statics = defaultdict(AttribDict)

        reader = csv.reader(
            filter(lambda row: row[0]!='#', self._file),
            delimiter=' ',
            skipinitialspace=True)

        for row in reader:
            _code = self._parse_code(row[0])

            # CAPUAF does not implement body-wave statics
            statics[_code]['body_wave_Z'] = 0.
            statics[_code]['body_wave_R'] = 0.

            statics[_code]['surface_wave_Z'] = float(row[10])
            statics[_code]['surface_wave_R'] = float(row[10])
            statics[_code]['surface_wave_T'] = float(row[11])

        return statics


def parse_station_codes(filename):
    """ Parses CAPUAF-style weight file
    """
    return WeightParser(filename).parse_weights().keys()



def remove_unused_stations(dataset, filename):
    """ Removes any stations not listed in CAPUAF weight file and stations with
        all zero weights
    """
    weights = WeightParser(filename).parse_weights()

    used = []
    unused = []

    for stream in dataset:
        id = stream.id

        if id not in weights:
            unused+=[id]
            continue

        if weights[id][1]==weights[id][2]==\
           weights[id][3]==weights[id][4]==weights[id][5]==0.:
            unused+=[id]
        else:
            used+=[id]

    if len(used)==0:
        warn(
            "No data selected. Please check that the current dataset matches "
            "the station codes in column 1 of the CAPUAF weight file and that "
            "columns 3-7 contain at least one nonzero weight.")

    for id in unused:
        dataset.remove(id)


#
# These rupture and rise time utilties can be used to generate source-time
# functions which match those defined in the cap Perl wrapper.  For use in
# conjunction with mtuq/util/wavelets.py:EarthquakeTrapezoid
#

def Trapezoid(magnitude=None):
    if magnitude is None:
        raise ValueError

    return EarthquakeTrapezoid(
        rupture_time=cap_rupture_time(magnitude),
        rise_time=cap_rise_time(magnitude))


def cap_rupture_time(Mw):
    rupture_time = np.floor(
        10.**(0.5*Mw - 2.5) + 0.5)

    if rupture_time < 1.:
        rupture_time = 1.

    if rupture_time > 9.:
        rupture_time = 9.

    return rupture_time


def cap_rise_time(Mw):
    return 0.5*cap_rupture_time(Mw)



#
# the following functions help reproduce cap signal processing functionality.
# See also mtuq/process_data/cap.py
#

def taper(array, taper_fraction=0.3, inplace=True):
    """ Reproduces CAP taper behavior. Similar to obspy Tukey?
    """
    if inplace:
        array = array
    else:
        array = np.copy(array)
    f = taper_fraction
    M = int(round(f*len(array)))
    I = np.linspace(0.,1.,M)
    taper = 0.5*(1-np.cos(np.pi*I))
    array[:M] *= taper
    array[-1:-M-1:-1] *= taper
    if not inplace:
        return array


#
# the following functions are used in benchmark comparisons between CAP and
# MTUQ synthetics. See tests/benchmark_cap_mtuq.py
#

def get_synthetics_cap(dummy_bw, dummy_sw, path, event_name):
    synthetics_bw = deepcopy(dummy_bw)
    synthetics_sw = deepcopy(dummy_sw)

    for stream in synthetics_bw:
        for trace in stream:
            trace.weight = 1.
            component = trace.meta.channel[-1].upper()

            if component == 'Z':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 9)
                trace_cap = obspy.read(filename, format='sac')[0]

            elif component == 'R':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 7)
                trace_cap = obspy.read(filename, format='sac')[0]

            else:
                continue

            if trace.meta.npts == trace_cap.meta.npts:
                trace.data = trace_cap.data
                # convert from cm/s to m/s
                trace.data *= 1.e-2

            else:
                stream.remove(trace)

    for stream in synthetics_sw:
        for trace in stream:
            trace.weight = 1.
            component = trace.meta.channel[-1].upper()

            if component == 'Z':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 5)
                trace.data = obspy.read(filename, format='sac')[0].data

            if component == 'R':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 3)
                trace.data = obspy.read(filename, format='sac')[0].data

            if component == 'T':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 1)
                trace.data = obspy.read(filename, format='sac')[0].data

            # convert from cm/s to m/s
            trace.data *= 1.e-2

    return synthetics_bw, synthetics_sw


def get_data_cap(dummy_bw, dummy_sw, path, event_name):
    data_bw = deepcopy(dummy_bw)
    data_sw = deepcopy(dummy_sw)

    for stream in data_bw:
        for trace in stream:
            trace.weight = 1.
            component = trace.meta.channel[-1].upper()

            if component == 'Z':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 8)
                trace_cap = obspy.read(filename, format='sac')[0]

            elif component == 'R':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 6)
                trace_cap = obspy.read(filename, format='sac')[0]

            else:
                continue

            if trace.meta.npts == trace_cap.meta.npts:
                trace.data = trace_cap.data
                # convert from cm/s to m/s
                trace.data *= 1.e-2
            else:
                stream.remove(trace)

    for stream in data_sw:
        for trace in stream:
            trace.weight = 1.
            component = trace.meta.channel[-1].upper()

            if component == 'Z':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 4)
                trace.data = obspy.read(filename, format='sac')[0].data

            elif component == 'R':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 2)
                trace.data = obspy.read(filename, format='sac')[0].data

            elif component == 'T':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 0)
                trace.data = obspy.read(filename, format='sac')[0].data

            # convert from cm/s to m/s
            trace.data *= 1.e-2


    return data_bw, data_sw



def get_synthetics_mtuq(dummy_bw, dummy_sw, greens_bw, greens_sw, mt,
                        apply_shifts=True):

    synthetics_bw = deepcopy(dummy_bw)
    synthetics_sw = deepcopy(dummy_sw)

    for synthetics, greens in ((synthetics_bw, greens_bw), (synthetics_sw, greens_sw)):
        for i in range(len(synthetics)):
            components = []
            for trace in synthetics[i]:
                components += [trace.meta.channel[-1]]
            greens[i]._set_components(components)
            dummy = greens[i].get_synthetics(mt)
            for trace in synthetics[i]:
                trace.weight = 1.
                component = trace.meta.channel[-1].upper()
                trace.data = dummy.select(component=component)[0].data

                if apply_shifts:
                    apply_magnitude_dependent_shift(trace, mt.magnitude())

    return synthetics_bw, synthetics_sw



def apply_magnitude_dependent_shift(trace, Mw):
    """ This type of time-shift arises from the idiosyncratic way CAP
      implements source-time function convolution. CAP's "conv" function
      results in systematic magnitude-dependent shifts between origin times
      and arrival times. This is arguably a bug. We include this function
      to allow benchmark comparisons between MTUQ synthetics (which normally
      lack such shifts) and CAP synthetics.
    """
    # the amount of the time-shift is half the sum of the earthquake's rupture
    # time and rise time, as given by relations in the CAP Perl wrapper
    t_offset = (cap_rupture_time(Mw) + cap_rise_time(Mw))/2.

    dt = trace.stats.delta
    nt = int(t_offset/dt)

    # shift trace to right by nt samples
    trace.data[nt:] = trace.data[:-nt]
    trace.data[:nt] = 0.


def compare_cap_mtuq(cap_bw_, cap_sw_, mtuq_bw_, mtuq_sw_,
                     bw_tol=np.inf, sw_tol=1.e-2, norm=2):
    """ Checks whether CAP and MTUQ synthetics agree within the specified
      tolerances

      Even with the magnitude-dependent time-shift correction described above,
      CAP and MTUQ synthetics will not match perfectly because the correction
      is made only after tapering

      For body wave windows, the time-shift correction is large relative to the
      window length, and the tapering-related mismatch will usually be
      pronounced. Thus, checking body waves is turned off by default
    """
    # keep track of number of mismatches
    count = 0

    for cap_bw, cap_sw, mtuq_bw, mtuq_sw in zip(
        cap_bw_, cap_sw_, mtuq_bw_, mtuq_sw_):

        if bw_tol < np.inf:
            maxval = 0.
            for bw in mtuq_bw:
                maxval = max(maxval, abs(bw.data).max())

            for bw1, bw2 in zip(cap_bw, mtuq_bw):
                dt = bw1.stats.delta
                e = np.linalg.norm((bw1.data-bw2.data)*dt, norm)
                e *= dt/maxval
                if e > bw_tol:
                    print(((
                        "Discrepancy between CAP and MTUQ synthetics\n"+
                        "  category:  body waves\n"+
                        "  error:     %e\n"+
                        "  threshold: %e\n") %
                        (e, bw_tol)))

                    count += 1


        if sw_tol < np.inf:
            maxval = 0.
            for sw in mtuq_sw:
                maxval = max(maxval, abs(sw.data).max())

            for sw1, sw2 in zip(cap_sw, mtuq_sw):
                dt = sw1.stats.delta
                e = np.linalg.norm((sw1.data-sw2.data)*dt, norm)
                e *= dt/maxval
                if e > sw_tol:
                    print(((
                        "Discrepancy between CAP and MTUQ synthetics\n"+
                        "  category:  surface waves\n"+
                        "  id:        %s\n"+
                        "  error:     %e\n"+
                        "  threshold: %e\n") %
                        (sw1.id, e, sw_tol)))

                    count += 1

    if count > 0:
        raise Exception(
            "Discrepancy between CAP and MTUQ synthetics")
