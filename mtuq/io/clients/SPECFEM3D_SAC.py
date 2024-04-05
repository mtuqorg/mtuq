
import glob
import obspy
import numpy as np

from obspy.core import Stream
from os.path import join
from mtuq.greens_tensor.SPECFEM3D import GreensTensor 
from mtuq.io.clients.base import Client as ClientBase
from mtuq.util.signal import resample


EXT_MT= [
    'Z.Mrr',
    'Z.Mtt',
    'Z.Mpp',
    'Z.Mrt',
    'Z.Mrp',
    'Z.Mtp',
    'R.Mrr',
    'R.Mtt',
    'R.Mpp',
    'R.Mrt',
    'R.Mrp',
    'R.Mtp',
    'T.Mrr',
    'T.Mtt',
    'T.Mpp',
    'T.Mrt',
    'T.Mrp',
    'T.Mtp',
    ]

EXT_FORCE = [
    'R.Fe',
    'T.Fe',
    'Z.Fe',
    'R.Fs',
    'T.Fs',
    'Z.Fs',
    'R.Fz',
    'T.Fz',
    'Z.Fz',
    ]



class Client(ClientBase):
    """ SPECFEM3D Green's tensor client

    .. rubric:: Usage

    To instantiate a database client, supply a path or url:

    .. code::

        from mtuq.io.clients.SPECFEM3D_SAC import Client
        db = Client(path_or_url)

    Then the database client can be used to generate GreensTensors:

    .. code::

        greens_tensors = db.get_greens_tensors(stations, origin)


    .. note::

    """

    def __init__(self, path_or_url=None, model=None, 
                 include_mt=True, include_force=False,
                 units='km'):

        self.path = path_or_url

        if not model:
            model = path_or_url
        self.model = model

        self.include_mt = include_mt
        self.include_force = include_force

        if not units:
            pass
        elif units.lower() not in ['','m','km']:
            raise ValueError
            units = units.lower()
        self.units = units


    def get_greens_tensors(self, stations=[], origins=[], verbose=False):
        """ Extracts Green's tensors

        Returns a ``GreensTensorList`` in which each element corresponds to a
        (station, origin) pair from the given lists

        .. rubric :: Input arguments

        ``stations`` (`list` of `mtuq.Station` objects)

        ``origins`` (`list` of `mtuq.Origin` objects)

        ``verbose`` (`bool`)

        """
        return super(Client, self).get_greens_tensors(stations, origins, verbose)


    def _get_greens_tensor(self, station=None, origin=None):
        stream = Stream()

        # read data
        stream = Stream()
        stream.id = station.id

        # optionally, append depth subdirectory to path
        path = self.path
        if hasattr(origin, 'depth_in_m'):
           if self.units=='m':
               subdir = str(int(np.ceil(origin.depth_in_m)))
           elif self.units=='km':
               subdir = str(int(np.ceil(origin.depth_in_m/1000.)))
           else:
               raise ValueError
           path = join(path, subdir)

        # check if Green's functions exist for given station code
        if _exists(path+'/'+station.id+'.*.sac'):
            prefix = station.id

        else:
            print('No Green\'s functions found for "%s"\n\n'
                  'Trying other codes instead:'
                  % station.id)

            prefix = _try_wildcards(path, station)

        if self.include_mt:
            for suffix in EXT_MT:
                trace = obspy.read(
                    path+'/'+prefix+'.'+suffix+'.sac', format='sac')[0]
                trace.stats.channel = suffix
                trace.stats._component = suffix[0]
                stream += trace

        if self.include_force:
            for suffix in EXT_FORCE:
                trace = obspy.read(
                    path+'/'+prefix+'.'+suffix+'.sac', format='sac')[0]
                trace.stats.channel = suffix
                trace.stats._component = suffix[0]
                stream += trace


        # what are the start and end times of the data?
        t1_new = float(station.starttime)
        t2_new = float(station.endtime)
        dt_new = float(station.delta)

        # what are the start and end times of the Green's function?
        # (SPECFEM3D_GLOBE defines these begin and end times relative to
        # peak excitation time)
        t1_old = float(stream[0].stats.sac['b'])
        t2_old = float(stream[0].stats.sac['e'])
        dt_old = float(stream[0].stats.delta)
        t1_old += float(origin.time)
        t2_old += float(origin.time)

        for trace in stream:
            # resample Green's functions
            data_old = trace.data
            data_new = resample(data_old, t1_old, t2_old, dt_old,
                                          t1_new, t2_new, dt_new)
            trace.data = data_new
            trace.stats.starttime = t1_new
            trace.stats.delta = dt_new
            trace.stats.npts = len(data_new)

        tags = [
            'model:%s' % self.model,
            'solver:%s' % 'SPECFEM3D',
             ]

        return GreensTensor(traces=[trace for trace in stream], 
            station=station, origin=origin, tags=tags,
            include_mt=self.include_mt, include_force=self.include_force)



def _exists(wildcard):
    if len(glob.glob(wildcard)) > 0:
        return True
    else:
        return False

def _try_wildcards(path, station):

    wildcards = [ 
        station.network+'.'+station.station+'.*',
        '*'+'.'+station.station+'.*',
        ] 

    for wildcard in wildcards:
        print('  "%s"' % wildcard)
        if _exists(path+'/'+wildcard+'.*.sac'):
            print()
            return wildcard
    else:
        raise FileNotFoundError()


