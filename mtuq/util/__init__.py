
from copy import deepcopy
from functools import reduce
from math import ceil, floor
from matplotlib import colors
from obspy import UTCDateTime
from os.path import abspath, join
from retry import retry

import copy
import csv
import json
import time
import numpy as np
import obspy
import re
import uuid
import warnings
import zipfile


# Python2/3 compatibility
try:
    from urllib import URLopener
except ImportError:
    from urllib.request import URLopener

from six import string_types


class AttribDict(obspy.core.util.attribdict.AttribDict):
    """ For storing trace attributes (see `mtuq.graphics.attrs`)
    """
    pass


def asarray(x):
    """ NumPy array typecast
    """
    try:
        return np.array(x, dtype=np.float64, ndmin=1, copy=False)
    except:
        return np.array(x, dtype=np.float64, ndmin=1)



def gather2(comm, array):
    """ Gathers 2-D NumPy arrays and combines along first dimension

    For very large numbers of elements, provides improved performance over 
    `gather` by using the lower-level function `Gatherv`
    """
    from mpi4py import MPI


    if not isinstance(array, np.ndarray):
        raise NotImplementedError

    if array.dtype=="float32":
        mpi_type = MPI.FLOAT

    elif array.dtype=="float64":
        mpi_type = MPI.DOUBLE

    else:
        raise NotImplementedError

    if array.ndim!=2:
        raise NotImplementedError

    # TODO - how to enforce same ncol for all processes?
    nrow, ncol = array.shape


    # start by defining the memory block sizes and create receiving buffer
    sendcounts = np.array(comm.gather(np.size(array), root=0))

    if comm.rank == 0:
        recvbuf = np.empty(sum(sendcounts), dtype=array.dtype)
    else:
        recvbuf = None

    comm.Gatherv(
        sendbuf=(array, mpi_type),
        recvbuf=(recvbuf, sendcounts, None, mpi_type),
        root=0)

    if comm.rank == 0:
        return recvbuf.reshape(int(len(recvbuf)/ncol),ncol)
    else:
        return


def is_mpi_env():
    try:
        import mpi4py
    except ImportError:
        return False

    try:
        import mpi4py.MPI
    except ImportError:
        return False

    if mpi4py.MPI.COMM_WORLD.Get_size()>1:
        return True
    else:
        return False


def iterable(obj):
    """ Simple list typecast
    """
    if isinstance(obj, string_types):
        return [obj]

    if issubclass(type(obj), dict) or issubclass(type(obj), AttribDict):
        return [obj]

    try:
        (item for item in obj)
    except TypeError:
        obj = [obj]
    finally:
        return obj


def merge_dicts(*dicts):
   merged = {}
   for dict in dicts:
      merged.update(dict)
   return merged


def product(*arrays):
    return reduce((lambda x, y: x * y), arrays)


def remove_list(list1, list2):
    """ Removes all items of list2 from list1
    """
    for item in list2:
        try:
            list1.remove(item)
        except ValueError:
            pass
    return list1


def replace(string, *args):
    narg = len(args)

    iarg = 0
    while iarg < narg:
        string = re.sub(args[iarg], args[iarg+1], string)
        iarg += 2
    return string


def basepath():
    """ MTUQ base directory
    """
    import mtuq
    return abspath(join(mtuq.__path__[0], '..'))


def fullpath(*args):
    """ Prepends MTUQ base diretory to given path
    """
    return join(basepath(), *args)


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):

        from mtuq import Station
        if isinstance(obj, Station):
            # don't write out SAC metadata (too big)
            if hasattr(obj, 'sac'):
                obj = deepcopy(obj)
                obj.pop('sac', None)

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, AttribDict):
            return obj.__dict__
        if issubclass(type(obj), AttribDict):
            return obj.__dict__
        if isinstance(obj, UTCDateTime):
            return str(obj)

        return super(JSONEncoder, self).default(obj)


def save_json(filename, data):
    if type(data) == AttribDict:
        data = {key: data[key] for key in data}

    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, cls=JSONEncoder, ensure_ascii=False, indent=4)


def timer(func):
    """ Decorator for measuring execution time
    """
    def timed_func(*args, **kwargs):
        if kwargs.get('timed', True):
            start_time = time.time()
            output = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            print('\n    Elapsed time (s): %.3f\n' % elapsed_time)
            return output
        else:
            return func(*args, **kwargs)

    return timed_func


def to_rgb(color):
    """ Converts matplotlib color to red-green-blue tuple
    """
    return 255*asarray(colors.to_rgba(color)[:3])


def unzip(filename):
    parts = filename.split('.')
    if parts[-1]=='zip':
        dirname = '.'.join(parts[:-1])
    else:
        dirname = filename
        filename += '.zip'

    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(dirname)
    zip_ref.close()

    return dirname


def warn(*args, **kwargs):
    if is_mpi_env():
        from mpi4py import MPI
        if MPI.COMM_WORLD.rank==0:
           warnings.warn(*args, **kwargs)
    else:
       warnings.warn(*args, **kwargs)


@retry(Exception, tries=4, delay=2, backoff=2)
def urlopen_with_retry(url, filename):
    opener = URLopener()
    opener.retrieve(url, filename)


def url2uuid(url):
    """ Converts a url to a uuid string
    """
    namespace = uuid.NAMESPACE_URL
    name = url
    return uuid.uuid5(namespace, name)


class Null(object):
    """ Always and reliably does nothing
    """
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getitem__(self, *args, **kwargs):
        return None

    def __nonzero__(self):
        return False


class ProgressCallback(object):
    """ Displays progress messages

    Displays messages at a regular interval (specified as a percentagage of the
    total number of iterations), when called from a loop or other iterative
    procedure
    """
    def __init__(self, start, stop, percent):

        start = int(round(start))
        stop = int(round(stop))

        assert (0 <= start)
        assert (start <= stop)

        if percent==0.:
            self.iter = start
            self.next_iter = float("inf")
            return
        elif 0 < percent <= 1:
            percent = 1
        elif 1 < percent <= 50.:
            percent = int(ceil(percent))
        elif 50 < percent <= 100:
            percent = 50
        else:
            raise ValueError

        self.start = start
        self.stop = stop
        self.percent = percent
        self.msg_interval = percent/100.*stop
        self.msg_count = int(100./percent*start/stop)
        self.iter = start
        self.next_iter = self.msg_count * self.msg_interval


    def __call__(self):
        if self.iter >= self.next_iter:
            print("  about %d percent finished" % (self.msg_count*self.percent))
            self.msg_count += 1
            self.next_iter = self.msg_count * self.msg_interval
        self.iter += 1


def dataarray_idxmin(da, warnings=True):
    """ idxmin helper function
    """
    # something similar to this has now been implemented in a beta version 
    # of xarray
    da = da.where(da==da.min(), drop=True).squeeze()
    if da.size > 1:
        if warnings:
            warn("No unique global minimum\n")
        while da.size > 1:
            da = da[0]
    return da.coords


def dataarray_idxmax(da, warnings=True):
    """ idxmax helper function
    """
    # something similar to this has now been implemented in a beta version 
    # of xarray
    da = da.where(da==da.max(), drop=True).squeeze()
    if da.size > 1:
        if warnings:
            warn("No unique global maximum\n")
        while da.size > 1:
            da = da[0]
    return da.coords


def defaults(kwargs, defaults):
    for key in defaults:
        if key not in kwargs:
           kwargs[key] = defaults[key]

