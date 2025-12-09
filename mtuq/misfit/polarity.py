
import numpy as np

import mtuq
from mtuq.util import AttribDict, Null, iterable, warn
from mtuq.misfit.waveform.level2 import _to_array, _type
from obspy.taup import TauPyModel
from mtuq.util.signal import m_to_deg


class PolarityMisfit(object):
    """ Polarity misfit function

    .. rubric:: Usage

    Evaluating polarity misfit is a two-step procedure.

    First, the user supplies parameters such as the method used to calculate
    predicted polarities (see below for detailed parameter descriptions):

    .. code::

        polarity_misfit = PolarityMisfit(**parameters)

    Second, the user invokes the misfit function:

    .. code::

        values = polarity_misfit(data, greens, sources)

    During misfit evaluation

    - observed polarities are collected from the `data` argument, which can be
      either a list of integers or a `Dataset` with observed polarity values
      attached (see convention below for more information)

    - predicted polarities are calculated from the `greens` argument (see 
      parameter descriptions below for more information)

    - a NumPy array of length `len(sources)` is returned, giving the number of
      mismatches between observed and predicted


    .. rubric:: Parameters

    ``method`` (`str`)

    - ``'taup'``
      Calculate polarities using Taup-P

    - ``'FK_metadata'``
      Read polarities from FK database

    - ``'CPS_metadata'``
      Read polarities from CPS database

    - ``'waveform'``
      Determine polarity from full-waveform synthetics (not implemented yet)


    .. rubric:: Other input arguments that may be required, depending on the above

    ``taup_model`` (`str`): Name of built-in ObsPy TauP model or path to custom
    ObsPy TauP model, required for `method=taup`

    ``FK_database`` (`str`): Path to FK database, required for for `method=FK_metadata`.

    .. note:: 

      *Convention* : Positive vertical first motions are encoded as +1 and
      negative vertical first motions are encoded as -1. Unpicked or 
      indeterminate first motions can be encoded as 0.

    """

    def __init__(self,
        method='taup',
        taup_model='ak135',
        FK_database=None,
        FK_model=None,
        CPS_database=None,
        CPS_model=None,
        **parameters):

        if not method:
            raise Exception('Undefined parameter: method')

        self.method = method
        self.taup_model = taup_model
        self.FK_database = FK_database
        self.FK_model = FK_model
        self.CPS_database = CPS_database
        self.CPS_model = CPS_model


        #
        # check parameters
        #
        if self.method == 'taup':
            assert self.taup_model is not None
            self._taup = TauPyModel(self.taup_model)

        elif self.method == 'FK_metadata':
            assert self.FK_database is not None
            assert exists(self.FK_database)
            if self.FK_model is None:
                self.FK_model = basename(self.FK_database)

        elif self.method == 'CPS_metadata':
            assert self.CPS_database is not None
            assert exists(self.CPS_database)
            if self.CPS_model is None:
                self.CPS_model = basename(self.CPS_database)

        else:
            raise TypeError('Bad parameter: method')


    def __call__(self, data, greens, sources, progress_handle=Null(),
            set_attributes=False):

        # check input arguments
        _check(greens, self.method)

 
        #
        # evaluate misfit
        #
        observed = self.get_observed(data)
        predicted = self.get_predicted(greens, sources)

        values = np.abs(predicted - observed)/2.

        # mask unpicked
        mask = (observed != 0).astype(int)
        values = np.dot(values, mask)

        # returns a NumPy array of shape (len(sources), 1)
        return values.reshape(len(values), 1)


    def get_observed(self, data):
        """ Extracts observed polarities from data
        """

        if isinstance(data, mtuq.Dataset):
            observed = np.array([_get_polarity(stream) for stream in data])

        elif isinstance(data, list):
            observed = np.array(data)

        elif isinstance(data, np.ndarray):
            observed = data

        else:
            raise TypeError

        return observed


    def get_predicted(self, greens, sources):
        """ Calculates predicted polarities
        """

        if type(sources) == mtuq.MomentTensor:
            sources = sources.as_vector().reshape((1,6))
            _calculate = _polarities_mt

        elif type(sources) == mtuq.Force:
            raise NotImplementedError

        elif _type(sources.dims) == 'MomentTensor':
            sources = _to_array(sources)
            _calculate = _polarities_mt

        elif _type(sources.dims) == 'Force':
            raise NotImplementedError
        else:
            raise TypeError

        if self.method=='taup':
            takeoff_angles = _takeoff_angles_taup(self._taup, greens)

            azimuths = _get_azimuths(greens)

            predicted = _calculate(sources, takeoff_angles, azimuths)

        elif self.method=='FK_metadata':
            takeoff_angles = _takeoff_angles_FK(self.FK_database, greens)

            azimuths = _get_azimuths(greens)

            predicted = _calculate(sources, takeoff_angles, azimuths)

        elif self.method == 'CPS_metadata':
            takeoff_angles = _takeoff_angles_CPS(self.CPS_database, greens)

            azimuths = _get_azimuths(greens)

            predicted = _calculate(sources, takeoff_angles, azimuths)

        elif self.method=='waveform':
            raise NotImplementedError

        return predicted


    def collect_attributes(self, data, greens):
        """ Collect polarity-related attributes (used for beachball plots)
        """

        if self.method=='taup':
            takeoff_angles = _takeoff_angles_taup(
                self._taup, greens)

        elif self.method=='FK_metadata':
            takeoff_angles = _takeoff_angles_FK(
                self.FK_database, greens)

        elif self.method=='CPS_metadata':
            takeoff_angles = _takeoff_angles_CPS(
                self.CPS_database, greens)

        observed = self.get_observed(data)

        attrs_list = []
        for _i, greens_tensor in enumerate(greens):
            attrs = AttribDict()

            try:
                attrs.azimuth = greens_tensor.azimuth
                attrs.distance_in_m = greens_tensor.distance_in_m
            except:
                pass
            try:
                attrs.network = greens_tensor.station.network
                attrs.station = greens_tensor.station.station
                attrs.location = greens_tensor.station.location
                attrs.latitude = greens_tensor.station.latitude
                attrs.longitude = greens_tensor.station.longitude
            except:
                pass
            try:
                attrs.takeoff_angle = takeoff_angles[_i]
            except:
                pass
            try:
                attrs.polarity = observed[_i]
            except:
                pass

            attrs_list += [attrs]

        return attrs_list


    def description(self):
        _description = '\n'.join([
            f'    Misfit function type:\n    {type(self).__name__}\n',
            f'    Misfit function method:\n    {self.method}\n',
            ])
        return _description




def _takeoff_angles_taup(taup, greens):
    """ Calculates takeoff angles from Tau-P model
    """

    takeoff_angles = np.zeros(len(greens))

    for _i, greens_tensor in enumerate(greens):

        depth_in_km = greens_tensor.origin.depth_in_m/1000.
        distance_in_deg = m_to_deg(greens_tensor.distance_in_m)

        takeoff_angles[_i] = _takeoff_angle_taup(
            taup, depth_in_km, distance_in_deg)

    return takeoff_angles


def _takeoff_angle_taup(taup, depth_in_km, distance_in_deg):

    arrivals = taup.get_travel_times(
        depth_in_km, distance_in_deg, phase_list=['p', 'P'])
    
    if len(arrivals) == 0:
        raise Exception
    else:
        sorted_arrivals = sorted(arrivals, key=lambda arrival: arrival.time)
        return sorted_arrivals[0].takeoff_angle


def _polarities_mt(mt_array, takeoff, azimuth):

    n1,n2 = mt_array.shape
    if n2!= 6:
        raise Exception('Inconsistent dimensions')

    n3,n4 = len(takeoff), len(azimuth)
    if n3!=n4:
        raise Exception('Inconsistent dimensions')

    # prepare arrays
    polarities = np.zeros((n1,n3))
    drc = np.empty((n1, 3))
    takeoff = np.deg2rad(takeoff)
    azimuth = np.deg2rad(azimuth)

    for _i, angles in enumerate(zip(takeoff, azimuth)):
        sth = np.sin(angles[0])
        cth = np.cos(angles[0])
        sphi = np.sin(angles[1])
        cphi = np.cos(angles[1])

        drc[:, 0] = sth*cphi
        drc[:, 1] = sth*sphi
        drc[:, 2] = cth

        # Aki & Richards 2ed, p. 108, eq. 4.88
        cth = mt_array[:, 0]*drc[:, 2]*drc[:, 2] +\
              mt_array[:, 1]*drc[:, 0]*drc[:, 0] +\
              mt_array[:, 2]*drc[:, 1]*drc[:, 1] +\
           2*(mt_array[:, 3]*drc[:, 0]*drc[:, 2] -
              mt_array[:, 4]*drc[:, 1]*drc[:, 2] -
              mt_array[:, 5]*drc[:, 0]*drc[:, 1])

        polarities[cth > 0, _i] = +1
        polarities[cth < 0, _i] = -1

    return polarities


def _polarities_force(force_array, takeoff_array, azimuth_array):
    raise NotImplementedError


def _get_azimuths(greens):
    return np.array([stream.azimuth for stream in greens])


#
# input argument checking
#

def _model_type(greens):
    try:
        solver = greens[0].attrs.solver
    except:
        solver = 'Unknown'

    if solver in ('AxiSEM', 'FK', 'CPS', 'syngine'):
        model_type = '1D model'

    elif solver in ('SPECFEM3D', 'SPECFEM3D_GLOBE'):
        model_type = '3D model'

    else:
        model_type = 'Unknown model'

    return model_type


def _check(greens, method):
    return

    #model = _model_type(greens)
    #method = _method_type(method)

    #if model != method:
    #    print()
    #    print('  Possible inconsistency?')
    #    print('  Green''s functions are from: %s' % model)
    #    print('  Polarities are from: %s' % method)
    #    print()


