

Imports="""
import os
import numpy as np

from mtuq import read, open_db, download_greens
from mtuq.event import Origin
from mtuq.graphics import plot_data_greens2, plot_beachball, plot_misfit_dc
from mtuq.grid import DoubleCoupleGridRegular
from mtuq.grid_search import grid_search
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid


"""


Docstring_DetailedAnalysis="""
if __name__=='__main__':
    #
    # Performs detailed analysis involving
    #
    # - grid search over all moment tensor parameters, including magnitude
    # - separate body wave, Rayleigh wave and Love wave data categories
    # - data variance estimation and likelihood analysis
    #
    #
    # Generates figures of
    #
    # - maximum likelihood surfaces
    # - marginal likelihood surfaces
    # - data misfit surfaces
    # - "variance reduction" surfaces
    # - geographic variation of time shifts
    # - geographic variation of amplitude ratios
    #
    #
    # USAGE
    #   mpirun -n <NPROC> python DetailedAnalysis.py
    #   
    #
    # This is the most complicated example. For simpler ones, see
    # SerialGridSearch.DoubleCouple.py or GridSearch.FullMomentTensor.py
    #
    # For ideas on applying this type of analysis to entire sets of events,
    # see github.com/rmodrak/mtbench
    #

"""


Docstring_GridSearch_DoubleCouple="""
if __name__=='__main__':
    #
    # Carries out grid search over 64,000 double couple moment tensors
    #
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.DoubleCouple.py
    #
    # For a simpler example, see SerialGridSearch.DoubleCouple.py, 
    # which runs the same inversion in serial
    #

"""


Docstring_GridSearch_DoubleCoupleMagnitudeDepth="""
if __name__=='__main__':
    #
    # Carries out grid search over source orientation, magnitude, and depth
    #   
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.DoubleCouple+Magnitude+Depth.py
    #
    # For simpler examples, see SerialGridSearch.DoubleCouple.py or
    # GridSearch.FullMomentTensor.py
    #   

"""


Docstring_GridSearch_DoubleCoupleMagnitudeHypocenter="""
if __name__=='__main__':
    #
    # Carries out grid search over source orientation, magnitude, and hypocenter 
    #   
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.DoubleCouple+Magnitude+Hypocenter.py
    #

    #
    # 1D Green's functions will be downloaded from a remote server, which can 
    # take a very long time. Any subsequent runs will generally be much faster.
    # A local Green's function database can be even faster still (see online 
    # documentation for more information).
    #
    # More meaningful results could be obtained using 3D Green's functions and
    # a phase misfit function, but 3D databases are too large for remote 
    # hosting. 
    #
    # If you are just trying things out for the first time, consider running 
    # one of the other examples instead.  Beacause they require fewer Green's
    # functions, all the other examples have faster and more consistent 
    # runtimes.
    #

"""


Docstring_GridSearch_FullMomentTensor="""
if __name__=='__main__':
    #
    # Carries out grid search over all moment tensor parameters
    #
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.FullMomentTensor.py
    #   

"""


Docstring_SerialGridSearch_DoubleCouple="""
if __name__=='__main__':
    #
    # Carries out grid search over 64,000 double couple moment tensors
    #
    # USAGE
    #   python SerialGridSearch.DoubleCouple.py
    #
    # A typical runtime is about 60 seconds. For faster results try 
    # GridSearch.DoubleCouple.py, which runs the same inversion in parallel
    #

"""


Docstring_WaveformsPolarities="""
if __name__=='__main__':
    #
    # Joint waveform and polarity grid search over all moment tensor parameters
    #
    # USAGE
    #   mpirun -n <NPROC> python Waveforms+Polarities.py
    #   
    # For a simpler example, see SerialGridSearch.DoubleCouple.py
    #

"""


Docstring_TestGridSearch_DoubleCouple="""
if __name__=='__main__':
    #
    # Grid search integration test
    #
    # This script is similar to examples/SerialGridSearch.DoubleCouple.py,
    # except here we use a coarser grid, and at the end we assert that the test
    # result equals the expected result
    #
    # The compare against CAP/FK:
    #
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1/1/10/10/10 -R0/0/0/0/0/360/0/90/-180/180 20090407201255351
    #
    # Note however that CAP uses a different method for defining regular grids
    #

"""


Docstring_TestGridSearch_DoubleCoupleMagnitudeDepth="""
if __name__=='__main__':
    #
    # Grid search integration test
    #
    # For speed, the script tests the iterative machinery for depth
    # searches in a somewhat degenerate case, using only a length one 
    # container
    #
    # For a depth search in the more usual sense, see
    # examples/GridSearch.DoubleCouple+Magnitude+Depth.py
    #

"""


Docstring_TestGraphics="""
if __name__=='__main__':
    #
    # Tests data, synthetics and beachball plotting utilities
    #

    #
    # The idea is for a test that runs very quickly, suitable for CI testing;
    # eventually we may more detailed tests to tests/graphics
    #

    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib
"""



Docstring_TestMisfit="""
if __name__=='__main__':
    #
    # Checks the correctness of the fast (optimized) misfit function
    # implementations against a simple pure Python implementation.
    # These implementations correspond to:
    #
    #   level=0: simple pure Python
    #   level=1: fast pure Python
    #   level=2: fast Python/C
    #
    # In running the test in our environment, we observe that the two pure 
    # Python implementations agree almost exactly.  On the other hand, the
    # pure Python and Python/C results differ by as much as 0.1 percent, 
    # presumably as a result of differences in the way that floating-point
    # error accumulates in the sum over residuals. Further work is required to 
    # understand this better
    #
    # Possibly relevant is the fact that C extensions are compiled with
    # `-Ofast` flag, as specified in `setup.py`.
    #
    # Note that the `level` keyword argument does not correspond
    # at all to C compiler optimization flags.  For example, the NumPy binaries
    # called by the simple pure Python misfit function are probably compiled 
    # using a nonzero optimization level?
    #


"""


Docstring_Gallery="""
if True:
    #
    # Creates example data structures
    #
    # Rather than being executed as a script, this code is designed to be
    # imported.  After importing this module, users can access the example data
    # and functions listed in __all__
    #
    # Note that some I/O and data processing are involved in creating the
    # example data, so importing this module may take significantly longer than
    # other modules
    #
    
    __all__ = [
        'process_bw'
        'process_bw'
        'misfit_bw',
        'misfit_bw',
        'data_bw',
        'data_sw',
        'greens_bw',
        'greens_sw',
        'stations',
        'origin',
        ]
"""


Docstring_BenchmarkCAP="""
if __name__=='__main__':
    #
    # Given seven "fundamental" moment tensors, generates MTUQ synthetics and
    # compares with corresponding CAP/FK synthetics
    #
    # Before running this script, it is necessary to unpack the CAP/FK 
    # synthetics using data/tests/unpack.bash
    #
    # This script is similar to examples/SerialGridSearch.DoubleCouple.py,
    # except here we consider only seven grid points rather than an entire
    # grid, and here the final plots are a comparison of MTUQ and CAP/FK 
    # synthetics rather than a comparison of data and synthetics
    #
    # Because of the idiosyncratic way CAP implements source-time function
    # convolution, it's not expected that CAP and MTUQ synthetics will match 
    # exactly. CAP's "conv" function results in systematic magnitude-
    # dependent shifts between origin times and arrival times. We deal with 
    # this by applying magnitude-dependent time-shifts to MTUQ synthetics 
    # (which normally lack such shifts) at the end of the benchmark. Even with
    # this correction, the match will not be exact because CAP applies the 
    # shifts before tapering and MTUQ after tapering. The resulting mismatch 
    # will usually be apparent in body-wave windows, but not in surface-wave 
    # windows
    #
    # Note that CAP works with dyne,cm and MTUQ works with N,m, so to make
    # comparisons we convert CAP output from the former to the latter
    #
    # The CAP/FK synthetics used in the comparison were generated by 
    # uafseismo/capuaf:46dd46bdc06e1336c3c4ccf4f99368fe99019c88
    # using the following commands
    #
    # source #0 (explosion):
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R0/1.178/90/45/90 20090407201255351
    #
    # source #1 (on-diagonal)
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R-0.333/0.972/90/45/90 20090407201255351
    #
    # source #2 (on-diagonal)
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R-0.333/0.972/45/90/180 20090407201255351
    #
    # source #3 (on-diagonal):
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R-0.333/0.972/45/90/0 20090407201255351
    #
    # source #4 (off-diagonal):
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R0/0/90/90/90 20090407201255351
    #
    # source #5 (off-diagonal):
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R0/0/90/0/0 20090407201255351
    #
    # source #6 (off-diagonal):
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R0/0/0/90/180 20090407201255351
    #

"""


ArgparseDefinitions="""
    # by default, the script runs with figure generation and error checking
    # turned on
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_checks', action='store_true')
    parser.add_argument('--no_figures', action='store_true')
    args = parser.parse_args()
    run_checks = (not args.no_checks)
    run_figures = (not args.no_figures)

"""


Paths_BenchmarkCAP="""
    from mtuq.util.cap import\\
        get_synthetics_cap, get_synthetics_mtuq,\\
        get_data_cap, compare_cap_mtuq


    # the following directories correspond to the moment tensors in the list 
    # "sources" below
    paths = []
    paths += [fullpath('data/tests/benchmark_cap/20090407201255351/0')]
    paths += [fullpath('data/tests/benchmark_cap/20090407201255351/1')]
    paths += [fullpath('data/tests/benchmark_cap/20090407201255351/2')]
    paths += [fullpath('data/tests/benchmark_cap/20090407201255351/3')]
    paths += [fullpath('data/tests/benchmark_cap/20090407201255351/4')]
    paths += [fullpath('data/tests/benchmark_cap/20090407201255351/5')]
    paths += [fullpath('data/tests/benchmark_cap/20090407201255351/6')]

"""


PathsComments="""
    #
    # We will investigate the source process of an Mw~4 earthquake using data
    # from a regional seismic array
    #
"""


Paths_Syngine="""
    path_data=    fullpath('data/examples/20090407201255351/*.[zrt]')
    path_weights= fullpath('data/examples/20090407201255351/weights.dat')
    event_id=     '20090407201255351'
    model=        'ak135'

"""


Paths_AxiSEM="""
    path_greens= '/home/rmodrak/data/ak135f_scak-2s'
    path_data=    fullpath('data/examples/20090407201255351/*.[zrt]')
    path_weights= fullpath('data/examples/20090407201255351/weights.dat')
    event_id=     '20090407201255351'
    model=        'ak135f_scak-2s'

"""


Paths_FK="""
    path_greens=  fullpath('data/tests/benchmark_cap/greens/scak')
    path_data=    fullpath('data/examples/20090407201255351/*.[zrt]')
    path_weights= fullpath('data/examples/20090407201255351/weights.dat')
    event_id=     '20090407201255351'
    model=        'scak'

"""

Paths_SPECFEM3D_SGT="""
    path_greens = fullpath('data/examples/SPECFEM3D_SGT/greens/socal3D')
    path_data   = fullpath('data/examples/SPECFEM3D_SGT/data/*.[zrt]')
    path_weights= fullpath('data/examples/SPECFEM3D_SGT/weights.dat')
    event_id    = 'evt11056825'
    model       = 'socal3D'
    taup_model  = 'ak135'

"""



DataProcessingComments="""
    #
    # Body and surface wave measurements will be made separately
    #
"""


DataProcessingDefinitions="""
    process_bw = ProcessData(
        filter_type='Bandpass',
        freq_min= 0.1,
        freq_max= 0.333,
        pick_type='taup',
        taup_model=model,
        window_type='body_wave',
        window_length=15.,
        capuaf_file=path_weights,
        )

    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=0.025,
        freq_max=0.0625,
        pick_type='taup',
        taup_model=model,
        window_type='surface_wave',
        window_length=150.,
        capuaf_file=path_weights,
        )

"""


MisfitComments="""
    #
    # For our objective function, we will use a sum of body and surface wave
    # contributions
    #
"""


MisfitDefinitions="""
    misfit_bw = Misfit(
        norm='L2',
        time_shift_min=-2.,
        time_shift_max=+2.,
        time_shift_groups=['ZR'],
        normalize=True,
        )

    misfit_sw = Misfit(
        norm='L2',
        time_shift_min=-10.,
        time_shift_max=+10.,
        time_shift_groups=['ZR','T'],
        normalize=True,
        )

"""

WaveformsPolaritiesMisfit="""
    #
    # We will jointly evaluate waveform differences and polarities
    #

    misfit_bw = WaveformMisfit(
        norm='L2',
        time_shift_min=-2.,
        time_shift_max=+2.,
        time_shift_groups=['ZR'],
        )

    misfit_sw = WaveformMisfit(
        norm='L2',
        time_shift_min=-10.,
        time_shift_max=+10.,
        time_shift_groups=['ZR','T'],
        )

    polarity_misfit = PolarityMisfit(
        taup_model=model)


    #
    # Observed polarities can be attached to the data or passed through a 
    # user-supplied dictionary or list in which +1 corresopnds to positive 
    # first motion, -1 to negative first moation, and 0 to indeterminate or
    # unpicked
    #

    polarities_dict = {
        "BMR":   0,
        "DIV":  +1,
        "EYAK": +1,
        "PAX":  +1,
        "SWD":  +1,
        "TRF":  -1,
        "PMR":  -1,
        "AVAL": +1,
        "BIGB": -1,
        "BLAK": +1,
        "DEVL": +1,
        "HEAD": +1,
        "KASH": -1,
        "LSKI": -1,
        "LSUM": +1,
        "MPEN":  0,
        "NSKI": +1,
        "PERI": +1,
        "SOLD":  0,
        "TUPA":  1,
        }


"""


WeightsComments="""
    #
    # User-supplied weights control how much each station contributes to the
    # objective function
    #
"""


WeightsDefinitions="""
    station_id_list = parse_station_codes(path_weights)

"""


OriginComments="""
    #
    # Origin time and location will be fixed. For an example in which they 
    # vary, see examples/GridSearch.DoubleCouple+Magnitude+Depth.py
    #
    # See also Dataset.get_origins(), which attempts to create Origin objects
    # from waveform metadata
    #
"""


OriginDefinitions="""
    origin = Origin({
        'time': '2009-04-07T20:12:55.000000Z',
        'latitude': 61.454200744628906,
        'longitude': -149.7427978515625,
        'depth_in_m': 33033.599853515625,
        })

"""


OriginDefinitions_SPECFEM3D_SGT="""
    origin = Origin({
        'time': '2019-07-04T18:39:44.0000Z',
        'latitude': 35.601333,
        'longitude': -117.597,
        'depth_in_m': 2810.0,
        'id': 'evt11056825'
        })

"""


OriginsComments="""
    #
    # We will search over a range of locations about the catalog origin
    #

"""


Origins_Depth="""
    catalog_origin = Origin({
        'time': '2009-04-07T20:12:55.000000Z',
        'latitude': 61.454200744628906,
        'longitude': -149.7427978515625,
        'depth_in_m': 33033.599853515625,
        })

    depths = np.array(
         # depth in meters
        [25000., 30000., 35000., 40000.,                    
         45000., 50000., 55000., 60000.])

    origins = []
    for depth in depths:
        origins += [catalog_origin.copy()]
        setattr(origins[-1], 'depth_in_m', depth)


"""



Origins_Hypocenter="""
    catalog_origin = Origin({
        'time': '2009-04-07T20:12:55.000000Z',
        'latitude': 61.454200744628906,
        'longitude': -149.7427978515625,
        'depth_in_m': 33033.599853515625,
        })

    from mtuq.util.math import lat_lon_tuples
    tuples = lat_lon_tuples(
        center_lat=catalog_origin.latitude,
        center_lon=catalog_origin.longitude,
        spacing_in_m=1000.,
        npts_per_edge=4,
        )

    origins = []
    for lat, lon in tuples:
        origins += [catalog_origin.copy()]
        setattr(origins[-1], 'latitude', lat)
        setattr(origins[-1], 'longitude', lon)

        # use best depth from DC+Depth search
        setattr(origins[-1], 'depth_in_m', 45000.)
        
"""


MisfitDefinitions_DetailedExample="""
    misfit_bw = Misfit(
        norm='L2',
        time_shift_min=-2.,
        time_shift_max=+2.,
        time_shift_groups=['ZR'],
        normalize=False,
        )

    misfit_rayleigh = Misfit(
        norm='L2',
        time_shift_min=-10.,
        time_shift_max=+10.,
        time_shift_groups=['ZR'],
        normalize=False,
        )

    misfit_love = Misfit(
        norm='L2',
        time_shift_min=-10.,
        time_shift_max=+10.,
        time_shift_groups=['T'],
        normalize=False,
        )

"""


Grid_DoubleCouple="""
    #
    # Next, we specify the moment tensor grid and source-time function
    #

    grid = DoubleCoupleGridRegular(
        npts_per_axis=40,
        magnitudes=[4.5])

    wavelet = Trapezoid(
        magnitude=4.5)

"""


Grid_DoubleCoupleMagnitude="""
    #
    # Next, we specify the moment tensor grid and source-time function
    #

    magnitudes = np.array(
         # moment magnitude (Mw)
        [4.3, 4.4, 4.5,     
         4.6, 4.7, 4.8]) 

    grid = DoubleCoupleGridRegular(
        npts_per_axis=20,
        magnitudes=magnitudes)

    wavelet = Trapezoid(
        magnitude=4.5)

"""


Grid_FullMomentTensor="""
    #
    # Next, we specify the moment tensor grid and source-time function
    #

    grid = FullMomentTensorGridSemiregular(
        npts_per_axis=10,
        magnitudes=[4.4, 4.5, 4.6, 4.7])

    wavelet = Trapezoid(
        magnitude=4.5)

"""


Grid_TestDoubleCoupleMagnitudeDepth="""
    #
    # Next, we specify the moment tensor grid and source-time function
    #

    grid = DoubleCoupleGridRegular(
        npts_per_axis=5,
        magnitudes=[4.4, 4.5, 4.6, 4.7])

    wavelet = Trapezoid(
        magnitude=4.5)

"""+OriginDefinitions+"""
    depths = np.array(
         # depth in meters
        [34000])

    origins = []
    for depth in depths:
        origin.depth = depth
        origins += [origin.copy()]
"""


Grid_TestGraphics="""
    from mtuq import MomentTensor

    mt = MomentTensor(
        1.e16 * np.sqrt(1./3.)*np.array([1., 1., 1., 0., 0., 0.])) # explosion

    mt_dict = {
       'rho':1.,'v':0.,'w':3/8*np.pi,'kappa':0.,'sigma':0.,'h':0.}

    wavelet = Trapezoid(
        magnitude=4.5)
"""


Grid_BenchmarkCAP="""
    #
    # Next we specify the source parameter grid
    #

    magnitude = 4.5
    moment = 10.**(1.5*magnitude + 9.1) # units: N-m

    sources = []
    for array in [
       # Mrr, Mtt, Mpp, Mrt, Mrp, Mtp
       np.sqrt(1./3.)*np.array([1., 1., 1., 0., 0., 0.]), # explosion
       np.array([1., 0., 0., 0., 0., 0.]), # source 1 (on-diagonal)
       np.array([0., 1., 0., 0., 0., 0.]), # source 2 (on-diagonal)
       np.array([0., 0., 1., 0., 0., 0.]), # source 3 (on-diagonal)
       np.sqrt(1./2.)*np.array([0., 0., 0., 1., 0., 0.]), # source 4 (off-diagonal)
       np.sqrt(1./2.)*np.array([0., 0., 0., 0., 1., 0.]), # source 5 (off-diagonal)
       np.sqrt(1./2.)*np.array([0., 0., 0., 0., 0., 1.]), # source 6 (off-diagonal)
       ]:

        sources += [MomentTensor(np.sqrt(2)*moment*array)]

    wavelet = Trapezoid(
        magnitude=magnitude)

"""


Main_GridSearch="""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD


    #
    # The main I/O work starts now
    #

    if comm.rank==0:
        print('Reading data...\\n')
        data = read(path_data, format='sac', 
            event_id=event_id,
            station_id_list=station_id_list,
            tags=['units:m', 'type:velocity']) 


        data.sort_by_distance()
        stations = data.get_stations()


        print('Processing data...\\n')
        data_bw = data.map(process_bw)
        data_sw = data.map(process_sw)


        print('Reading Greens functions...\\n')
        greens = download_greens(stations, origin, model)

        print('Processing Greens functions...\\n')
        greens.convolve(wavelet)
        greens_bw = greens.map(process_bw)
        greens_sw = greens.map(process_sw)


    else:
        stations = None
        data_bw = None
        data_sw = None
        greens_bw = None
        greens_sw = None


    stations = comm.bcast(stations, root=0)
    data_bw = comm.bcast(data_bw, root=0)
    data_sw = comm.bcast(data_sw, root=0)
    greens_bw = comm.bcast(greens_bw, root=0)
    greens_sw = comm.bcast(greens_sw, root=0)


    #
    # The main computational work starts now
    #

    if comm.rank==0:
        print('Evaluating body wave misfit...\\n')

    results_bw = grid_search(
        data_bw, greens_bw, misfit_bw, origin, grid)

    if comm.rank==0:
        print('Evaluating surface wave misfit...\\n')

    results_sw = grid_search(
        data_sw, greens_sw, misfit_sw, origin, grid)

"""


Main1_SerialGridSearch_DoubleCouple="""
    #
    # The main I/O work starts now
    #

    print('Reading data...\\n')
    data = read(path_data, format='sac',
        event_id=event_id,
        station_id_list=station_id_list,
        tags=['units:m', 'type:velocity']) 


    data.sort_by_distance()
    stations = data.get_stations()


    print('Processing data...\\n')
    data_bw = data.map(process_bw)
    data_sw = data.map(process_sw)


    print('Reading Greens functions...\\n')
    greens = download_greens(stations, origin, model)


    print('Processing Greens functions...\\n')
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw)
    greens_sw = greens.map(process_sw)

"""


Main2_SerialGridSearch_DoubleCouple="""
    #
    # The main computational work starts now
    #

    print('Evaluating body wave misfit...\\n')
    results_bw = grid_search(data_bw, greens_bw, misfit_bw, origin, grid)

    print('Evaluating surface wave misfit...\\n')
    results_sw = grid_search(data_sw, greens_sw, misfit_sw, origin, grid)

"""


Main_TestGridSearch_DoubleCoupleMagnitudeDepth="""
    #
    # The main I/O work starts now
    #

    print('Reading data...\\n')
    data = read(path_data, format='sac', 
        event_id=event_id,
        station_id_list=station_id_list,
        tags=['units:m', 'type:velocity']) 

    data.sort_by_distance()
    stations = data.get_stations()


    print('Processing data...\\n')
    data_bw = data.map(process_bw)
    data_sw = data.map(process_sw)


    print('Reading Greens functions...\\n')
    db = open_db(path_greens, format='FK', model=model)
    greens = db.get_greens_tensors(stations, origins)


    print('Processing Greens functions...\\n')
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw)
    greens_sw = greens.map(process_sw)


    #
    # The main computational work starts now
    #

    print('Evaluating body wave misfit...\\n')

    results_bw = grid_search(
        data_bw, greens_bw, misfit_bw, origins, grid, 0)

    print('Evaluating surface wave misfit...\\n')

    results_sw = grid_search(
        data_sw, greens_sw, misfit_sw, origins, grid, 0)


"""


Main_TestGraphics="""

    print('Reading data...\\n')
    data = read(path_data, format='sac',
        event_id=event_id,
        #station_id_list=station_id_list,
        tags=['units:m', 'type:velocity'])


    data.sort_by_distance()
    stations = data.get_stations()


    print('Processing data...\\n')
    data_bw = data.map(process_bw)
    data_sw = data.map(process_sw)

    print('Reading Greens functions...\\n')
    db = open_db(path_greens, format='FK', model=model)
    greens = db.get_greens_tensors(stations, origin)

    print('Processing Greens functions...\\n')
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw)
    greens_sw = greens.map(process_sw)


    #
    # Generate figures
    #

    print('Plot data (1 of 6)\\n')

    from mtuq.graphics import plot_waveforms2
    from mtuq.util import Null

    plot_waveforms2('graphics_test_1.png',
        data_bw, data_sw, Null(), Null(),
        stations, origin, header=False)


    print('Plot synthetics (2 of 6)\\n')

    synthetics_bw = greens_bw.get_synthetics(mt, components=['Z','R'])
    synthetics_sw = greens_sw.get_synthetics(mt, components=['Z','R','T'])


    plot_waveforms2('graphics_test_2.png',
        synthetics_bw, synthetics_sw, Null(), Null(),
        stations, origin, header=False)


    print('Plot synthetics (3 of 6)\\n')

    synthetics_bw = misfit_bw.collect_synthetics(data_bw, greens_bw, mt)
    synthetics_sw = misfit_sw.collect_synthetics(data_sw, greens_sw, mt)


    plot_waveforms2('graphics_test_3.png',
        synthetics_bw, synthetics_sw, Null(), Null(),
        stations, origin, header=False)


    print('Plot data and synthetics without header (4 of 6)\\n')

    plot_waveforms2('graphics_test_4.png',
        data_bw, data_sw, synthetics_bw, synthetics_sw,
        stations, origin, header=None)


    print('Plot data and synthetics with header (5 of 6)\\n')

    plot_data_greens2('graphics_test_5.png',
        data_bw, data_sw, greens_bw, greens_sw,
        process_bw, process_sw, misfit_bw, misfit_sw,
        stations, origin, mt, mt_dict)


    print('Plot explosion bechball (6 of 6)\\n')

    plot_beachball('graphics_test_6.png',
        mt, None, None)

    print('\\nFinished\\n')
"""



Main_TestMisfit="""
    #
    # The main computational work starts now
    #

    print('Evaluating body wave misfit...\\n')

    results_0 = misfit_bw(
        data_bw, greens_bw, grid, level=0)

    results_1 = misfit_bw(
        data_bw, greens_bw, grid, level=1)

    results_2 = misfit_bw(
        data_bw, greens_bw, grid, level=2)

    print('  optimization level:  0\\n', 
          '  argmin:  %d\\n' % results_0.argmin(), 
          '  min:     %e\\n\\n' % results_0.min())

    print('  optimization level:  1\\n', 
          '  argmin:  %d\\n' % results_1.argmin(), 
          '  min:     %e\\n\\n' % results_1.min())

    print('  optimization level:  2\\n', 
          '  argmin:  %d\\n' % results_2.argmin(), 
          '  min:     %e\\n\\n' % results_2.min())

    print('')

    assert results_0.argmin()==results_1.argmin()==results_2.argmin()


    print('Evaluating surface wave misfit...\\n')

    results_0 = misfit_sw(
        data_sw, greens_sw, grid, level=0)

    results_1 = misfit_sw(
        data_sw, greens_sw, grid, level=1)

    results_2 = misfit_sw(
        data_sw, greens_sw, grid, level=2)

    print('  optimization level:  0\\n', 
          '  argmin:  %d\\n' % results_0.argmin(), 
          '  min:     %e\\n\\n' % results_0.min())

    print('  optimization level:  1\\n', 
          '  argmin:  %d\\n' % results_1.argmin(), 
          '  min:     %e\\n\\n' % results_1.min())

    print('  optimization level:  2\\n', 
          '  argmin:  %d\\n' % results_2.argmin(), 
          '  min:     %e\\n\\n' % results_2.min())

    assert results_0.argmin()==results_1.argmin()==results_2.argmin()


"""


WrapUp_DetailedAnalysis="""
    if comm.rank==0:

        results_sum = results_bw + results_rayleigh + results_love

        #
        # Data variance estimation and likelihood analysis
        #

        # use minimum misfit as initial guess for maximum likelihood
        idx = results_sum.source_idxmin()
        best_mt = grid.get(idx)
        lune_dict = grid.get_dict(idx)
        mt_dict = best_mt.as_dict()


        print('Data variance estimation...\\n')

        sigma_bw = estimate_sigma(data_bw, greens_bw,
            best_mt, misfit_bw.norm, ['Z', 'R'],
            misfit_bw.time_shift_min, misfit_bw.time_shift_max)

        sigma_rayleigh = estimate_sigma(data_sw, greens_sw,
            best_mt, misfit_rayleigh.norm, ['Z', 'R'],
            misfit_rayleigh.time_shift_min, misfit_rayleigh.time_shift_max)

        sigma_love = estimate_sigma(data_sw, greens_sw,
            best_mt, misfit_love.norm, ['T'],
            misfit_love.time_shift_min, misfit_love.time_shift_max)

        stats = {'sigma_bw': sigma_bw,
                 'sigma_rayleigh': sigma_rayleigh,
                 'sigma_love': sigma_love}

        print('  Body wave variance:  %.3e' %
            sigma_bw**2)
        print('  Rayleigh variance:   %.3e' %
            sigma_rayleigh**2)
        print('  Love variance:       %.3e' %
            sigma_love**2)

        print()

        norm_bw = calculate_norm_data(data_bw, 
            misfit_bw.norm, ['Z', 'R'])
        norm_rayleigh = calculate_norm_data(data_sw, 
            misfit_rayleigh.norm, ['Z', 'R'])
        norm_love = calculate_norm_data(data_sw, 
            misfit_love.norm, ['T'])

        norms = {misfit_bw.norm+'_bw': norm_bw,
                 misfit_rayleigh.norm+'_rayleigh': norm_rayleigh,
                 misfit_love.norm+'_love': norm_love}


        print('Likelihood analysis...\\n')

        likelihoods, mle_lune, marginal_vw = likelihood_analysis(
            (results_bw, sigma_bw**2),
            (results_rayleigh, sigma_rayleigh**2),
            (results_love, sigma_love**2))

        # maximum likelihood vw surface
        likelihoods_vw = _product_vw(
            _likelihoods_vw_regular(results_bw, sigma_bw**2),
            _likelihoods_vw_regular(results_rayleigh, sigma_rayleigh**2),
            _likelihoods_vw_regular(results_love, sigma_love**2))

        # TODO - marginalize over the joint likelihood distribution instead
        marginals_vw = _product_vw(
            _marginals_vw_regular(results_bw, sigma_bw**2),
            _marginals_vw_regular(results_rayleigh, sigma_rayleigh**2),
            _marginals_vw_regular(results_love, sigma_love**2))


        #
        # Generate figures and save results
        #

        # only generate components present in the data
        components_bw = data_bw.get_components()
        components_sw = data_sw.get_components()

        # synthetics corresponding to minimum misfit
        synthetics_bw = greens_bw.get_synthetics(
            best_mt, components_bw, mode='map')

        synthetics_sw = greens_sw.get_synthetics(
            best_mt, components_sw, mode='map')


        # time shifts and other attributes corresponding to minimum misfit
        list_bw = misfit_bw.collect_attributes(
            data_bw, greens_bw, best_mt)

        list_rayleigh = misfit_rayleigh.collect_attributes(
            data_sw, greens_sw, best_mt)

        list_love = misfit_love.collect_attributes(
            data_sw, greens_sw, best_mt)

        list_sw = [{**list_rayleigh[_i], **list_love[_i]}
            for _i in range(len(stations))]

        dict_bw = {station.id: list_bw[_i] 
            for _i,station in enumerate(stations)}

        dict_rayleigh = {station.id: list_rayleigh[_i] 
            for _i,station in enumerate(stations)}

        dict_love = {station.id: list_love[_i] 
            for _i,station in enumerate(stations)}

        dict_sw = {station.id: list_sw[_i] 
            for _i,station in enumerate(stations)}



        print('Plotting observed and synthetic waveforms...\\n')

        plot_beachball(event_id+'FMT_beachball.png', 
            best_mt, stations, origin)

        plot_data_greens2(event_id+'FMT_waveforms.png',
            data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw,
            misfit_bw, misfit_rayleigh, stations, origin, best_mt, lune_dict)


        print('Plotting misfit surfaces...\\n')

        os.makedirs(event_id+'FMT_misfit', exist_ok=True)

        plot_misfit_lune(event_id+'FMT_misfit/bw.png', results_bw,
            title='Body waves')

        plot_misfit_lune(event_id+'FMT_misfit/rayleigh.png', results_rayleigh,
            title='Rayleigh waves')

        plot_misfit_lune(event_id+'FMT_misfit/love.png', results_love,
            title='Love waves')

        print()


        print('Plotting maximum likelihood surfaces...\\n')

        os.makedirs(event_id+'FMT_likelihood', exist_ok=True)

        plot_likelihood_lune(event_id+'FMT_likelihood/bw.png',
            results_bw, var=sigma_bw**2, 
            title='Body waves')

        plot_likelihood_lune(event_id+'FMT_likelihood/rayleigh.png',
            results_rayleigh, var=sigma_rayleigh**2, 
            title='Rayleigh waves')

        plot_likelihood_lune(event_id+'FMT_likelihood/love.png',
            results_love, var=sigma_love**2, 
            title='Love waves')

        _plot_lune(event_id+'FMT_likelihood/all.png',
            likelihoods_vw, colormap='hot_r',
            title='All data categories')

        print()


        print('Plotting marginal likelihood surfaces...\\n')

        os.makedirs(event_id+'FMT_marginal', exist_ok=True)

        plot_marginal_vw(event_id+'FMT_marginal/bw.png',
            results_bw, var=sigma_bw**2,
            title='Body waves')

        plot_marginal_vw(event_id+'FMT_marginal/rayleigh.png',
            results_rayleigh, var=sigma_rayleigh**2,
            title='Rayleigh waves')

        plot_marginal_vw(event_id+'FMT_marginal/love.png',
            results_love, var=sigma_love**2,
            title='Love waves')

        _plot_vw(event_id+'FMT_marginal/all.png',
            marginals_vw, colormap='hot_r',
            title='All data categories')

        print()


        print('Plotting variance reduction surfaces...\\n')

        os.makedirs(event_id+'FMT_variance_reduction', exist_ok=True)

        plot_variance_reduction_lune(event_id+'FMT_variance_reduction/bw.png',
            results_bw, norm_bw, title='Body waves',
            colorbar_label='Variance reduction (percent)')

        plot_variance_reduction_lune(event_id+'FMT_variance_reduction/rayleigh.png',
            results_rayleigh, norm_rayleigh, title='Rayleigh waves',
            colorbar_label='Variance reduction (percent)')

        plot_variance_reduction_lune(event_id+'FMT_variance_reduction/love.png',
            results_love, norm_love, title='Love waves', 
            colorbar_label='Variance reduction (percent)')

        print()


        print('Plotting tradeoffs...\\n')

        os.makedirs(event_id+'FMT_tradeoffs', exist_ok=True)

        plot_misfit_lune(event_id+'FMT_tradeoffs/orientation.png',
            results_sum, show_tradeoffs=True, title='Orientation tradeoffs')

        plot_magnitude_tradeoffs_lune(event_id+'FMT_tradeoffs/magnitude.png',
            results_sum, title='Magnitude tradeoffs', colorbar_label='Mw')

        print()


        print('Plotting time shift geographic variation...\\n')

        plot_time_shifts(event_id+'FMT_time_shifts/bw',
            list_bw, stations, origin)

        plot_time_shifts(event_id+'FMT_time_shifts/sw',
            list_sw, stations, origin)


        print('Plotting amplitude ratio geographic variation...\\n')

        plot_amplitude_ratios(event_id+'FMT_amplitude_ratios/bw',
            list_bw, stations, origin)

        plot_amplitude_ratios(event_id+'FMT_amplitude_ratios/sw',
            list_sw, stations, origin)


        print('\\nSaving results...\\n')

        # save best-fitting source
        os.makedirs(event_id+'FMT_solutions', exist_ok=True)

        save_json(event_id+'FMT_solutions/marginal_likelihood.json', marginal_vw)
        save_json(event_id+'FMT_solutions/maximum_likelihood.json', mle_lune)

        merged_dict = merge_dicts(lune_dict, mt_dict, origin,
            {'M0': best_mt.moment(), 'Mw': best_mt.magnitude()})

        save_json(event_id+'FMT_solutions/minimum_misfit.json', merged_dict)


        os.makedirs(event_id+'FMT_stats', exist_ok=True)

        save_json(event_id+'FMT_stats/data_variance.json', stats)
        save_json(event_id+'FMT_stats/data_norm.json', norms)


        # save stations and origins
        stations_dict = {station.id: station
            for _i,station in enumerate(stations)}

        save_json(event_id+'FMT_stations.json', stations_dict)
        save_json(event_id+'FMT_origins.json', {0: origin})


        # save time shifts and other attributes
        os.makedirs(event_id+'FMT_attrs', exist_ok=True)

        save_json(event_id+'FMT_attrs/bw.json', dict_bw)
        save_json(event_id+'FMT_attrs/sw.json', dict_sw)


        # save processed waveforms as binary files
        os.makedirs(event_id+'FMT_waveforms', exist_ok=True)

        data_bw.write(event_id+'FMT_waveforms/dat_bw')
        data_sw.write(event_id+'FMT_waveforms/dat_sw')

        synthetics_bw.write(event_id+'FMT_waveforms/syn_bw')
        synthetics_sw.write(event_id+'FMT_waveforms/syn_sw')


        # save misfit surfaces as netCDF files
        results_bw.save(event_id+'FMT_misfit/bw.nc')
        results_rayleigh.save(event_id+'FMT_misfit/rayleigh.nc')
        results_love.save(event_id+'FMT_misfit/love.nc')


        print('\\nFinished\\n')

"""




WrapUp_GridSearch="""

    if comm.rank==0:

        results = results_bw + results_sw

        #
        # Collect information about best-fitting source
        #

        # index of best-fitting moment tensor
        idx = results.source_idxmin()

        # MomentTensor object
        best_mt = grid.get(idx)

        # dictionary of lune parameters
        lune_dict = grid.get_dict(idx)

        # dictionary of Mij parameters
        mt_dict = best_mt.as_dict()

        merged_dict = merge_dicts(
            mt_dict, lune_dict, {'M0': best_mt.moment()},
            {'Mw': best_mt.magnitude()}, origin)


        #
        # Generate figures and save results
        #

        print('Generating figures...\\n')

        plot_data_greens2(event_id+'DC_waveforms.png',
            data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, 
            misfit_bw, misfit_sw, stations, origin, best_mt, lune_dict)


        plot_beachball(event_id+'DC_beachball.png',
            best_mt, stations, origin)


        plot_misfit_dc(event_id+'DC_misfit.png', results)


        print('Saving results...\\n')

        # save best-fitting source
        save_json(event_id+'DC_solution.json', merged_dict)


        # save misfit surface
        results.save(event_id+'DC_misfit.nc')


        print('\\nFinished\\n')

"""


WrapUp_GridSearch_DoubleCoupleMagnitudeDepth="""

    if comm.rank==0:

        results = results_bw + results_sw

        #
        # Collect information about best-fitting source
        #

        origin_idx = results.origin_idxmin()
        best_origin = origins[origin_idx]

        source_idx = results.source_idxmin()
        best_mt = grid.get(source_idx)

        # dictionary of lune parameters
        lune_dict = grid.get_dict(source_idx)

        # dictionary of Mij parameters
        mt_dict = best_mt.as_dict()

        merged_dict = merge_dicts(
            mt_dict, lune_dict, {'M0': best_mt.moment()},
            {'Mw': best_mt.magnitude()}, best_origin)


        #
        # Generate figures and save results
        #

        print('Generating figures...\\n')

        plot_data_greens2(event_id+'DC+Z_waveforms.png',
            data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, 
            misfit_bw, misfit_sw, stations, best_origin, best_mt, lune_dict)


        plot_misfit_depth(event_id+'DC+Z_misfit_depth.png', results, origins,
            title=event_id)


        plot_misfit_depth(event_id+'DC+Z_misfit_depth_tradeoffs.png', results, origins,
            show_tradeoffs=True, show_magnitudes=True, title=event_id)


        print('Saving results...\\n')

        # save best-fitting source
        save_json(event_id+'DC+Z_solution.json', merged_dict)


        # save origins
        origins_dict = {_i: origin 
            for _i,origin in enumerate(origins)}

        save_json(event_id+'DC+Z_origins.json', origins_dict)


        # save misfit surface
        results.save(event_id+'DC+Z_misfit.nc')


        print('\\nFinished\\n')

"""


WrapUp_SerialGridSearch_DoubleCouple="""

    results = results_bw + results_sw

    #
    # Collect information about best-fitting source
    #

    # index of best-fitting moment tensor
    idx = results.source_idxmin()

    # MomentTensor object
    best_mt = grid.get(idx)

    # dictionary of lune parameters
    lune_dict = grid.get_dict(idx)

    # dictionary of Mij parameters
    mt_dict = best_mt.as_dict()

    merged_dict = merge_dicts(
        mt_dict, lune_dict, {'M0': best_mt.moment()},
        {'Mw': best_mt.magnitude()}, origin)

    #
    # Generate figures and save results
    #

    print('Generating figures...\\n')

    plot_data_greens2(event_id+'DC_waveforms.png',
        data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, 
        misfit_bw, misfit_sw, stations, origin, best_mt, lune_dict)


    plot_beachball(event_id+'DC_beachball.png',
        best_mt, stations, origin)


    plot_misfit_dc(event_id+'DC_misfit.png', results)


    print('Saving results...\\n')

    # save best-fitting source
    save_json(event_id+'DC_solution.json', merged_dict)


    # save misfit surface
    results.save(event_id+'DC_misfit.nc')


    print('\\nFinished\\n')

"""


WrapUp_WaveformsPolarities="""
    if comm.rank==0:
        print('Evaluating polarity misfit...\\n')

    polarities = np.zeros(len(stations))
    for _i, station in enumerate(stations):
        polarities[_i] = polarities_dict[station.station]
        
    results_polarity = grid_search(
        polarities, greens_bw, polarity_misfit, origin, grid)
        

    if comm.rank==0:

        results = results_bw + results_sw

        # `grid` index corresponding to minimum misfit
        idx = results.source_idxmin()

        best_mt = grid.get(idx)
        lune_dict = grid.get_dict(idx)
        mt_dict = best_mt.as_dict()


        #
        # Generate figures and save results
        #

        print('Generating figures...\\n')

        plot_data_greens2(event_id+'FMT_waveforms.png',
            data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw,
            misfit_bw, misfit_sw, stations, origin, best_mt, lune_dict)

        plot_beachball(event_id+'FMT_beachball.png',
            best_mt, stations, origin)

        plot_misfit_lune(event_id+'FMT_misfit.png', results,
            title='Waveform Misfit')

        # generate polarity figures

        plot_misfit_lune(event_id+'FMT_misfit_polarity.png', results_polarity,
            show_best=False, title='Polarity Misfit', plot_type='scatter')

        # predicted polarities
        predicted = polarity_misfit.get_predicted(greens, best_mt)

        # station attributes
        attrs = polarity_misfit.collect_attributes(polarities, greens)

        plot_polarities(event_id+'FMT_beachball_polarity.png',
            polarities, predicted, attrs, origin, best_mt)

        print('\\nFinished\\n')

"""


WrapUp_TestGridSearch_DoubleCouple="""

    results = results_bw + results_sw

    # source corresponding to minimum misfit
    idx = results.source_idxmin()
    best_mt = grid.get(idx)
    lune_dict = grid.get_dict(idx)

    if run_figures:

        plot_data_greens2(event_id+'DC_waveforms.png',
            data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, 
            misfit_bw, misfit_sw, stations, origin, best_mt, lune_dict)

        plot_beachball(event_id+'DC_beachball.png',
            best_mt, None, None)


    if run_checks:
        def isclose(a, b, atol=1.e6, rtol=1.e-6):
            # the default absolute tolerance (1.e6) is several orders of 
            # magnitude less than the moment of an Mw=0 event

            for _a, _b, _bool in zip(
                a, b, np.isclose(a, b, atol=atol, rtol=rtol)):

                print('%s:  %.e <= %.1e + %.1e * %.1e' %\\
                    ('passed' if _bool else 'failed', abs(_a-_b), atol, rtol, abs(_b)))

            print('')

            return np.all(
                np.isclose(a, b, atol=atol, rtol=rtol))

        if not isclose(best_mt.as_vector(),
            np.array([
                 -5.282696e+15,
                  9.182139e+14,
                  4.364482e+15,
                  8.607951e+14,
                  4.534485e+15,
                  2.217397e+15,
                 ])
            ):
            raise Exception(
                "Grid search result differs from previous mtuq result")

        print('SUCCESS\\n')
"""


WrapUp_TestGridSearch_DoubleCoupleMagnitudeDepth="""
    results = results_bw + results_sw

    idx = results.source_idxmin()
    best_mt = grid.get(idx)

    if run_figures:
        filename = event_id+'_misfit_vs_depth.png'
        #misfit_vs_depth(filename, best_misfit)

    if run_checks:
        pass

    print('SUCCESS\\n')

"""


Main_BenchmarkCAP="""
    #
    # The benchmark starts now
    #

    print('Reading data...\\n')
    data = read(path_data, format='sac', 
        event_id=event_id,
        tags=['units:m', 'type:velocity']) 

    data.sort_by_distance()

    stations = data.get_stations()
    origin = data.get_origins()[0]


    print('Processing data...\\n')
    data_bw = data.map(process_bw)
    data_sw = data.map(process_sw)

    print('Reading Greens functions...\\n')
    db = open_db(path_greens, format='FK', model=model)
    greens = db.get_greens_tensors(stations, origin)

    print('Processing Greens functions...\\n')
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw)
    greens_sw = greens.map(process_sw)


    depth = int(origin.depth_in_m/1000.)+1
    name = '_'.join([model, str(depth), event_id])


    print('Comparing waveforms...')

    for _i, mt in enumerate(sources):
        print('  %d of %d' % (_i+1, len(sources)))

        cap_bw, cap_sw = get_synthetics_cap(
            data_bw, data_sw, paths[_i], name)

        mtuq_bw, mtuq_sw = get_synthetics_mtuq(
            data_bw, data_sw, greens_bw, greens_sw, mt)

        if run_figures:
            plot_waveforms2('cap_vs_mtuq_'+str(_i)+'.png',
                cap_bw, cap_sw, mtuq_bw, mtuq_sw, 
                stations, origin, trace_label_writer=None)

        if run_checks:
            compare_cap_mtuq(
                cap_bw, cap_sw, mtuq_bw, mtuq_sw)

    if run_figures:
        # "bonus" figure comparing how CAP processes observed data with how
        # MTUQ processes observed data
        mtuq_sw, mtuq_bw = data_bw, data_sw

        cap_sw, cap_bw = get_data_cap(
            data_bw, data_sw, paths[0], name)

        plot_waveforms2('cap_vs_mtuq_data.png',
            cap_bw, cap_sw, mtuq_bw, mtuq_sw, 
            stations, origin, trace_label_writer=None, normalize=False)

    print('\\nSUCCESS\\n')

"""


if __name__=='__main__':
    import os
    from mtuq.util import basepath, replace
    os.chdir(basepath())


    with open('examples/DetailedAnalysis.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(
            replace(
            Imports,
            'DoubleCoupleGridRegular',
            'FullMomentTensorGridSemiregular',
            'plot_misfit_dc',
            (
            'plot_misfit_lune,\\\n'+
            '    plot_likelihood_lune, plot_marginal_vw,\\\n'+
            '    plot_variance_reduction_lune, plot_magnitude_tradeoffs_lune,\\\n'+
            '    plot_time_shifts, plot_amplitude_ratios,\\\n'+
            '    likelihood_analysis, _likelihoods_vw_regular, _marginals_vw_regular,\\\n'+
            '    _plot_lune, _plot_vw, _product_vw\n'+
            'from mtuq.graphics.uq.vw import _variance_reduction_vw_regular'
            ),
            'from mtuq.misfit import Misfit',
            'from mtuq.misfit.waveform import Misfit, estimate_sigma, calculate_norm_data'
            ))
        file.write(Docstring_DetailedAnalysis)
        file.write(Paths_Syngine)
        file.write(DataProcessingComments)
        file.write(DataProcessingDefinitions)
        file.write(MisfitComments)
        file.write(MisfitDefinitions_DetailedExample)
        file.write(WeightsComments)
        file.write(WeightsDefinitions)
        file.write(
            replace(
            Grid_FullMomentTensor,
            'npts_per_axis=10',
            'npts_per_axis=12',
            ))
        file.write(OriginComments)
        file.write(OriginDefinitions)
        file.write(
            replace(
            Main_GridSearch,
            'surface wave',
            'Rayleigh wave',
            'results_sw',
            'results_rayleigh',
            'misfit_sw',
            'misfit_rayleigh',
            ))
        file.write("""
    if comm.rank==0:
        print('Evaluating Love wave misfit...\\n')

    results_love = grid_search(
        data_sw, greens_sw, misfit_love, origin, grid)\n"""
        )
        file.write(WrapUp_DetailedAnalysis)


    with open('examples/GridSearch.DoubleCouple.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(Imports)
        file.write(Docstring_GridSearch_DoubleCouple)
        file.write(PathsComments)
        file.write(Paths_Syngine)
        file.write(DataProcessingComments)
        file.write(DataProcessingDefinitions)
        file.write(MisfitComments)
        file.write(MisfitDefinitions)
        file.write(WeightsComments)
        file.write(WeightsDefinitions)
        file.write(Grid_DoubleCouple)
        file.write(OriginComments)
        file.write(OriginDefinitions)
        file.write(Main_GridSearch)
        file.write(WrapUp_GridSearch)


    with open('examples/GridSearch.DoubleCouple+Magnitude+Depth.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(
            replace(
            Imports,
            'plot_beachball',
            'plot_misfit_depth',
            ))
        file.write(Docstring_GridSearch_DoubleCoupleMagnitudeDepth)
        file.write(PathsComments)
        file.write(Paths_Syngine)
        file.write(DataProcessingComments)
        file.write(DataProcessingDefinitions)
        file.write(MisfitComments)
        file.write(MisfitDefinitions)
        file.write(WeightsComments)
        file.write(WeightsDefinitions)
        file.write(OriginsComments)
        file.write(Origins_Depth)
        file.write(Grid_DoubleCoupleMagnitude)
        file.write(
            replace(
            Main_GridSearch,
            'origin',
            'origins',
            ))
        file.write(WrapUp_GridSearch_DoubleCoupleMagnitudeDepth)


    with open('examples/GridSearch.DoubleCouple+Magnitude+Hypocenter.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(
            replace(
            Imports,
            'plot_beachball',
            'plot_misfit_latlon',
            ))
        file.write(Docstring_GridSearch_DoubleCoupleMagnitudeHypocenter)
        file.write(PathsComments)
        file.write(Paths_Syngine)
        file.write(DataProcessingComments)
        file.write(DataProcessingDefinitions)
        file.write(MisfitComments)
        file.write(MisfitDefinitions)
        file.write(WeightsComments)
        file.write(WeightsDefinitions)
        file.write(OriginsComments)
        file.write(Origins_Hypocenter)
        file.write(Grid_DoubleCoupleMagnitude)
        file.write(
            replace(
            Main_GridSearch,
            r'origin',
            r'origins',
            'Reading Greens functions...\\\\n',
            (
            'Reading Greens functions...\\\\n\\\\n'+
            '  Downloads can sometimes take as long as a few hours!\\\\n'
            ),
           r'download_greens\(stations, origin, model\)',
            'download_greens(stations, origin, model, verbose=True)',
            ))
        file.write(
            replace(
            WrapUp_GridSearch_DoubleCoupleMagnitudeDepth,
           r'DC\+Z',
            'DC+XY',
            'misfit_depth',
            'misfit_latlon',
            "title=event_id",
            "title=event_id, colorbar_label='L2 misfit'",
            'show_magnitudes=True, ',
            '',
            ))


    with open('examples/GridSearch.FullMomentTensor.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(
            replace(
            Imports,
            'DoubleCoupleGridRegular',
            'FullMomentTensorGridSemiregular',
            'plot_misfit_dc',
            'plot_misfit_lune',
            ))
        file.write(Docstring_GridSearch_FullMomentTensor)
        file.write(Paths_Syngine)
        file.write(DataProcessingComments)
        file.write(DataProcessingDefinitions)
        file.write(MisfitComments)
        file.write(MisfitDefinitions)
        file.write(WeightsComments)
        file.write(WeightsDefinitions)
        file.write(Grid_FullMomentTensor)
        file.write(OriginComments)
        file.write(OriginDefinitions)
        file.write(Main_GridSearch)
        file.write(
            replace(
            WrapUp_GridSearch,
            'DC',
            'FMT',
            'plot_misfit_dc',
            'plot_misfit_lune',
            ))


    with open('examples/SerialGridSearch.DoubleCouple.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(Imports)
        file.write(Docstring_SerialGridSearch_DoubleCouple)
        file.write(PathsComments)
        file.write(Paths_Syngine)
        file.write(DataProcessingComments)
        file.write(DataProcessingDefinitions)
        file.write(MisfitComments)
        file.write(MisfitDefinitions)
        file.write(WeightsComments)
        file.write(WeightsDefinitions)
        file.write(Grid_DoubleCouple)
        file.write(OriginComments)
        file.write(OriginDefinitions)
        file.write(Main1_SerialGridSearch_DoubleCouple)
        file.write(Main2_SerialGridSearch_DoubleCouple)
        file.write(WrapUp_SerialGridSearch_DoubleCouple)


    with open('examples/Waveforms+Polarities.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(
            replace(
            Imports,
            'DoubleCoupleGridRegular',
            'FullMomentTensorGridSemiregular',
            'plot_misfit_dc',
            'plot_misfit_lune',
            'plot_beachball',
            'plot_beachball, plot_polarities',
            'from mtuq.misfit import Misfit',
            'from mtuq.misfit import WaveformMisfit, PolarityMisfit',
            ))
        file.write(Docstring_WaveformsPolarities)
        file.write(Paths_Syngine)
        file.write(DataProcessingComments)
        file.write(DataProcessingDefinitions)
        file.write(WaveformsPolaritiesMisfit)
        file.write(WeightsComments)
        file.write(WeightsDefinitions)
        file.write(Grid_FullMomentTensor)
        file.write(OriginComments)
        file.write(OriginDefinitions)
        file.write(Main_GridSearch)
        file.write(WrapUp_WaveformsPolarities)


    #with open('tests/test_SPECFEM3D_SGT.py', 'w') as file:
    #    file.write("#!/usr/bin/env python\n")
    #    file.write(
    #        replace(
    #        Imports,
    #        'DoubleCoupleGridRegular',
    #        'FullMomentTensorGridSemiregular',
    #        'plot_misfit_dc',
    #        'plot_misfit_lune',
    #        ))
    #    file.write(Docstring_GridSearch_FullMomentTensor)
    #    file.write(Paths_SPECFEM3D_SGT)
    #    file.write(DataProcessingComments)
    #    file.write(
    #        replace(
    #        DataProcessingDefinitions,
    #        'taup_model=model',
    #        'taup_model=taup_model',
    #        ))
    #    file.write(MisfitComments)
    #    file.write(MisfitDefinitions)
    #    file.write(WeightsComments)
    #    file.write(WeightsDefinitions)
    #    file.write(Grid_FullMomentTensor)
    #    file.write(OriginComments)
    #    file.write(OriginDefinitions_SPECFEM3D_SGT)
    #    file.write(
    #        replace(
    #        Main_GridSearch,
    #        'greens = download_greens\(stations, origin, model\)',
    #        'db = open_db(path_greens, format=\'SPECFEM3D_SGT\', model=model)\n        '
    #       +'greens = db.get_greens_tensors(stations, origin)',
    #       ))
    #    file.write(
    #        replace(
    #        WrapUp_GridSearch,
    #        'DC',
    #        'FMT',
    #        'plot_misfit_dc',
    #        'plot_misfit_lune',
    #        ))


    with open('tests/test_grid_search_mt.py', 'w') as file:
        file.write(Imports)
        file.write(Docstring_TestGridSearch_DoubleCouple)
        file.write(ArgparseDefinitions)
        file.write(Paths_FK)
        file.write(
            replace(
            DataProcessingDefinitions,
            'pick_type=.*',
            "pick_type='FK_metadata',",
            'taup_model=.*,',
            'FK_database=path_greens,',
            ))
        file.write(MisfitDefinitions)
        file.write(
            replace(
            Grid_DoubleCouple,
            'npts.*,',
            'npts_per_axis=5,',
            ))
        file.write(WeightsDefinitions)
        file.write(OriginDefinitions)
        file.write(
            replace(
            Main1_SerialGridSearch_DoubleCouple,
           r'greens = download_greens\(stations, origin, model\)',
            'db = open_db(path_greens, format=\'FK\', model=model)\n    '
           +'greens = db.get_greens_tensors(stations, origin)',
            ))
        file.write(
            replace(
            Main2_SerialGridSearch_DoubleCouple,
            'origin, grid',
            'origin, grid, 0',
            ))
        file.write(WrapUp_TestGridSearch_DoubleCouple)


    with open('tests/test_grid_search_mt_depth.py', 'w') as file:
        file.write(
            replace(
            Imports,
            'plot_beachball',
            'plot_misfit_depth',
            ))
        file.write(Docstring_TestGridSearch_DoubleCoupleMagnitudeDepth)
        file.write(ArgparseDefinitions)
        file.write(Paths_FK)
        file.write(
            replace(
            DataProcessingDefinitions,
            'pick_type=.*',
            "pick_type='FK_metadata',",
            'taup_model=.*,',
            'FK_database=path_greens,',
            ))
        file.write(MisfitDefinitions)
        file.write(WeightsDefinitions)
        file.write(Grid_TestDoubleCoupleMagnitudeDepth)
        file.write(Main_TestGridSearch_DoubleCoupleMagnitudeDepth)
        file.write(WrapUp_TestGridSearch_DoubleCoupleMagnitudeDepth)


    with open('tests/test_misfit.py', 'w') as file:
        file.write(
            replace(
            Imports,
            ))
        file.write(Docstring_TestMisfit)
        file.write(Paths_FK)
        file.write(
            replace(
            DataProcessingDefinitions,
            'pick_type=.*',
            "pick_type='FK_metadata',",
            'taup_model=.*,',
            'FK_database=path_greens,',
            ))
        file.write(MisfitDefinitions)
        file.write(WeightsComments)
        file.write(WeightsDefinitions)
        file.write(
            replace(
            Grid_DoubleCouple,
            'npts.*,',
            'npts_per_axis=5,',
            ))
        file.write(OriginDefinitions)
        file.write(
            replace(
            Main1_SerialGridSearch_DoubleCouple,
           r'greens = download_greens\(stations, origin, model\)',
            'db = open_db(path_greens, format=\'FK\', model=model)\n    '
           +'greens = db.get_greens_tensors(stations, origin)',
            ))
        file.write(Main_TestMisfit)


    with open('tests/benchmark_cap_vs_mtuq.py', 'w') as file:
        file.write(
            replace(
            Imports,
            'Origin',
            'MomentTensor',
            'syngine',
            'fk',
            'plot_data_greens2',
            'plot_waveforms2',
            ))
        file.write(Docstring_BenchmarkCAP)
        file.write(ArgparseDefinitions)
        file.write(Paths_BenchmarkCAP)
        file.write(
            replace(
            Paths_FK,
            'data/examples/20090407201255351/weights.dat',
            'data/tests/benchmark_cap/20090407201255351/weights.dat',
            ))
        file.write(
            replace(
            DataProcessingDefinitions,
            'pick_type=.*',
            "pick_type='FK_metadata',",
            'taup_model=.*,',
            'FK_database=path_greens,',
            ))
        file.write(
            replace(
            MisfitDefinitions,
            'time_shift_max=.*',
            'time_shift_max=0.,',
            ))
        file.write(Grid_BenchmarkCAP)
        file.write(Main_BenchmarkCAP)


    with open('tests/test_graphics.py', 'w') as file:
        file.write(Imports)
        file.write(Docstring_TestGraphics)
        file.write(Paths_FK)
        file.write(
            replace(
            DataProcessingDefinitions,
            'pick_type=.*',
            "pick_type='FK_metadata',",
            'taup_model=.*,',
            'FK_database=path_greens,',
            ))
        file.write(MisfitDefinitions)
        file.write(OriginDefinitions)
        file.write(Grid_TestGraphics)
        file.write(Main_TestGraphics)


    with open('mtuq/util/gallery.py', 'w') as file:
        file.write(Imports)
        file.write(Docstring_Gallery)
        file.write(Paths_Syngine)
        file.write(DataProcessingDefinitions)
        file.write(MisfitDefinitions)
        file.write(
            replace(
            Grid_DoubleCouple,
            'npts.*',
            'npts_per_axis=10,',
            ))
        file.write(
            replace(
            Main1_SerialGridSearch_DoubleCouple,
            'print.*',
            '',
            ))


