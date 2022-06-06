
#
# graphics/big_beachball.py - first motion "beachball" plots with piercing point polarities
#

# Requires pygmt intstalled

# https://github.com/obspy/obspy/issues/2388

import os
import matplotlib.pyplot as pyplot
import numpy as np

from mtuq.event import MomentTensor
from mtuq.misfit.polarity import radiation_coef
from mtuq.util import warn


def beachball_pygmt(filename, mt, data, plot_all=False, display_plot=False):
    """ Moment tensor plot with stations polarities, implemented in PyGMT.

    .. rubric :: Input arguments


    ``filename`` (`str`):
    Name of output image file

    ``data`` (`mtuq.DataSet`):
    mtuq Dataset with valid sac header

    ``mt`` (`mtuq.MomentTensor`):
    Moment tensor object

    Warning : Implemented and tested with PyGMT v0.3.1, which is still in early developpment. It is possible that the code might break some time in the future, as nerwer versions of the code are rolled-out.

    This plotting function presuppose that the data is a valid mtuq.DataSet whith a sac header containing the station azimuth in the `az` key, the takeoff_angle in the key `user3` and the picked polarity in the key `user5`

    TODO: This function will benefit from having a different way of handling polarity than with the sac header used as of right now. This will enable easier usage for user that won't be using pysep to fetch the data.
    """
    import pygmt
    from pygmt.helpers import build_arg_string, use_alias
    # Format the moment tensor to the expected GMT input (lon, lat, depth, mrr, mtt, mff, mrt, mrf, mtf, exponent, lon2, lat2).

    focal_mechanism = np.append(np.append([0, 0, 10], mt.as_vector()), [25, 0, 0])

    # Initialize the pygmt plot with the beachball plot
    fig = pygmt.Figure()
    fig.meca(region=[-1.2, 1.2, -1.2, 1.2], projection='m0/0/5c', scale='9.9c',
             convention="mt", G='grey50', spec=focal_mechanism, N=False, M=True)

    # Create a list of traces containing only the picked stations
    picked_data = [sta[0] for sta in data if sta[0].stats.sac['user5'] != 0]
    # Create a list of traces containing the unpicked stations
    unpicked_data = [sta[0] for sta in data if sta[0].stats.sac['user5'] == 0]

    # List the theoretical and picked radiation coefficients for comparison purposes
    theoretical_rad_coef = np.asarray(
        [radiation_coef(mt.as_vector(), sta.stats.sac['user3'], sta.stats.sac['az']) for sta in picked_data])
    picked_rad_coef = np.asarray([sta.stats.sac['user5'] for sta in picked_data])

    # Place the piercing in 4 list if they are:
    # - picked up motion matching theoretical polarity
    # - picked down motion matching theoretical polarity
    # - picked up motion not matching theoretical polarity
    # - picked down motion not matching theoretical polarity

    up_matching_data = [sta for i, sta in enumerate(
        picked_data) if sta.stats.sac['user5'] == theoretical_rad_coef[i] and sta.stats.sac['user5'] == 1]
    down_matching_data = [sta for i, sta in enumerate(
        picked_data) if sta.stats.sac['user5'] == theoretical_rad_coef[i] and sta.stats.sac['user5'] == -1]
    down_unmatched_data = [sta for i, sta in enumerate(
        picked_data) if sta.stats.sac['user5'] != theoretical_rad_coef[i] and sta.stats.sac['user5'] == -1]
    up_unmatched_data = [sta for i, sta in enumerate(
            picked_data) if sta.stats.sac['user5'] != theoretical_rad_coef[i] and sta.stats.sac['user5'] == 1]

    # Define aliases for the pygmt function. Please refer to GMT 6.2.0 `polar` function documentation for a complete overview of all the available options and option details.
    @use_alias(
        D='offset',
        J='projection',
        M='size',
        S='symbol',
        E='ext_fill',
        G='comp_fill',
        F='background',
        Qe='ext_outline',
        Qg='comp_outline',
        Qf='mt_outline',
        T='station_labels'
    )
    def _pygmt_polar(trace_list, **kwargs):
        """ Wrapper around GMT polar function. ]
        Color arguments must be in {red, green, blue, black, white} and the symbol in {a,c,d,h,i,p,s,t,x} (see GMT polar function for reference).
        """

        # Define some default values to format the plot.
        defaultKwargs = {
            'D' : '0/0',
            'J' : 'm0/0/5c',
            'M' : '9.9c',
            'T' : '+f0.18c'
            }
        kwargs = { **defaultKwargs, **kwargs }
        print(kwargs)


        colorcodes = {
            "red": "255/0/0",
            "green": "0/255/0",
            "blue": "0/0/255",
            "white":"255, 255, 255",
            "black": "0/0/0"
        }
        for key in kwargs:
            try:
                kwargs[key]=colorcodes[kwargs[key]]
            except:
                pass

        tmp_filename='polar_temp.txt'
        with open(tmp_filename, 'w') as f:
            for sta in trace_list:
                if sta.stats.sac['user5'] == 1:
                    pol = '+'
                    f.write('{} {} {} {}'.format(sta.stats.network+'.'+sta.stats.station, sta.stats.sac['az'], sta.stats.sac['user3'], pol))
                    f.write('\n')

                elif sta.stats.sac['user5'] == -1:
                    pol = '-'
                    f.write('{} {} {} {}'.format(sta.stats.network+'.'+sta.stats.station, sta.stats.sac['az'], sta.stats.sac['user3'], pol))
                    f.write('\n')
                else:
                    pol = '0'
                    f.write('{} {} {} {}'.format(sta.stats.network+'.'+sta.stats.station, sta.stats.sac['az'], sta.stats.sac['user3'], pol))
                    f.write('\n')
                    print('Warning !: Station ', sta.stats.network+'.' +
                          sta.stats.station, ' has no picked polarity')

        arg_string = " ".join([tmp_filename, build_arg_string(kwargs)])
        with pygmt.clib.Session() as session:
            session.call_module('polar',arg_string)

        os.remove(tmp_filename)


    # plotting the 4 previously lists with different symbols and colors
    _pygmt_polar(up_matching_data, symbol='t0.40c', comp_fill='green')
    _pygmt_polar(down_matching_data, symbol='i0.40c', ext_fill='blue')
    _pygmt_polar(up_unmatched_data, symbol='t0.40c', comp_fill='red')
    _pygmt_polar(down_unmatched_data, symbol='i0.40c', ext_fill='red')
    # If `plot_all` is True, will plot the unpicked stations as black crosses over the beachball plot
    if not plot_all is False:
        _pygmt_polar(unpicked_data, symbol='x0.40c', comp_outline='black', ext_outline='black')

    # fig.show(dpi=300, method="external")
    fig.savefig(filename, show=display_plot)

