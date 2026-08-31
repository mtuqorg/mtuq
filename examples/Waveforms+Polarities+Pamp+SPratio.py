#!/usr/bin/env python

import os
import numpy as np

from mtuq import read, download_greens
from mtuq.event import Origin
from mtuq.graphics import plot_data_greens2, plot_beachball, plot_polarities,\
    plot_misfit_lune
from mtuq.grid import FullMomentTensorGridSemiregular
from mtuq.grid_search import grid_search
from mtuq.misfit import WaveformMisfit, PolarityPampSPampRatioMisfit,\
    polarities_pamp_spamp_ratio_from_dict
from mtuq.process_data import ProcessData
from mtuq.util import fullpath
from mtuq.util.cap import parse_station_codes, Trapezoid



if __name__=='__main__':
    #
    # Joint waveform, polarity, P-amplitude and S/P-amplitude-ratio grid
    # search over all moment tensor parameters
    #
    # USAGE
    #   mpirun -n <NPROC> python Waveforms+Polarities+Pamp+SPratio.py
    #
    # For an inversion that uses polarities but not amplitudes, see
    # Waveforms+Polarities.py
    #


    path_data=    fullpath('../pr348-minimal-rebuild_test/data/examples/20090407201255351/*.[zrt]')
    path_weights= fullpath('../pr348-minimal-rebuild_test/data/examples/20090407201255351/weights.dat')
    event_id=     '20090407201255351'
    model=        'ak135'


    #
    # Body and surface wave measurements will be made separately
    #

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
    # 
    # Polarity, P-amplitude, and S/P-amplitude-ratio misfit function
    #
    polarity_amplitude_misfit = PolarityPampSPampRatioMisfit(
        method='taup',
        taup_model=model,
        lambda_pol=1.,
        lambda_pamp=1.,
        lambda_sprat=1.,
        )


    #
    # Observed polarities: +1 for upward first motion, -1 for downward, and
    # 0 for indeterminate or unpicked
    #

    polarities_dict = {
        'BMR': +0,
        'DIV': +1,
        'EYAK': +1,
        'PAX': +1,
        'SWD': +1,
        'TRF': -1,
        'PMR': -1,
        'AVAL': +1,
        'BIGB': -1,
        'BLAK': +1,
        'DEVL': +1,
        'HEAD': +1,
        'KASH': -1,
        'LSKI': -1,
        'LSUM': +1,
        'MPEN': +0,
        'NSKI': +1,
        'PERI': +1,
        'SOLD': +0,
        'TUPA': +1,
        }


    #
    # Observed P amplitudes:
    # For each station, the amplitude envelope is computed, and its maximum
    # within a short window beginning at the theoretical direct-P arrival
    # is taken as the amplitude. The sign of the vertical component at that
    # same sample gives the polarity above, so a single measurement
    # supplies both observables.
    #
    # The window length is a measurement choice and should suit the
    # frequency band of the data being analyzed.
    #

    pamp_dict = {
        'BMR': +1.335232e-06,
        'DIV': +2.027537e-06,
        'EYAK': +1.653398e-06,
        'PAX': +7.862058e-07,
        'SWD': -1.814179e-06,
        'TRF': +2.008658e-06,
        'PMR': +5.345639e-07,
        'AVAL': -2.013863e-06,
        'BIGB': -9.375091e-06,
        'BLAK': +1.050444e-06,
        'DEVL': -3.230013e-06,
        'HEAD': -1.882761e-06,
        'KASH': -4.720312e-07,
        'LSKI': -3.104280e-06,
        'LSUM': -2.446641e-06,
        'MPEN': -3.688052e-06,
        'NSKI': -1.869809e-06,
        'PERI': +1.757306e-06,
        'SOLD': -1.602847e-06,
        'TUPA': -9.953246e-07,
        }


    #
    # Observed S/P amplitude ratios, computed from three-component
    # noise-corrected energies
    #
    #   rat = sqrt( [(S_R^2+S_T^2+S_Z^2) - (N_R^2+N_T^2+N_Z^2)] /
    #               [(P_R^2+P_T^2+P_Z^2) - (N_R^2+N_T^2+N_Z^2)] )
    #
    # The P and S windows begin at their corresponding theoretical direct
    # arrivals and end at a fraction of the S-P time; the noise window has
    # the same duration and precedes the event. The window fraction and the
    # signal-to-noise threshold used to accept a station are measurement
    # choices.
    #
    # These are linear ratios, not logarithms; the misfit function takes
    # log10 internally. If your measurements are recorded as log10, they are converted by passing
    # spamp_ratio_is_log10=True to polarities_pamp_spamp_ratio_from_dict.
    #
    spamp_dict = {
        'BMR': 5.208876,
        'DIV': 8.431348,
        'EYAK': 7.794872,
        'PAX': 3.216377,
        'SWD': 10.444677,
        'TRF': 2.907442,
        'PMR': 117.975275,
        'AVAL': 13.535989,
        'BIGB': 4.105131,
        'BLAK': 33.806172,
        'DEVL': 14.155526,
        'HEAD': 10.339749,
        'KASH': 112.384133,
        'LSKI': 5.269920,
        'LSUM': 7.586073,
        'MPEN': 5.245091,
        'NSKI': 13.103038,
        'PERI': 23.614676,
        'SOLD': 11.833848,
        'TUPA': 26.608782,
        }


    #
    # User-supplied weights control how much each station contributes to the
    # objective function
    #

    station_id_list = parse_station_codes(path_weights)


    #
    # Next, we specify the moment tensor grid and source-time function
    #

    grid = FullMomentTensorGridSemiregular(
        npts_per_axis=10,
        magnitudes=[4.4, 4.5, 4.6, 4.7])

    wavelet = Trapezoid(
        magnitude=4.5)


    #
    # Origin time and location will be fixed
    #

    origin = Origin({
        'time': '2009-04-07T20:12:55.000000Z',
        'latitude': 61.454200744628906,
        'longitude': -149.7427978515625,
        'depth_in_m': 33033.599853515625,
        })


    from mpi4py import MPI
    comm = MPI.COMM_WORLD


    #
    # The main I/O work starts now
    #

    if comm.rank==0:
        print('Reading data...\n')
        data = read(path_data, format='sac',
            event_id=event_id,
            station_id_list=station_id_list,
            tags=['units:m', 'type:velocity'])

        data.sort_by_distance()
        stations = data.get_stations()

        print('Processing data...\n')
        data_bw = data.map(process_bw)
        data_sw = data.map(process_sw)

        print('Reading Greens functions...\n')
        greens = download_greens(stations, origin, model)

        print('Processing Greens functions...\n')
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
        print('Evaluating body wave misfit...\n')

    results_bw = grid_search(
        data_bw, greens_bw, misfit_bw, origin, grid)

    if comm.rank==0:
        print('Evaluating surface wave misfit...\n')

    results_sw = grid_search(
        data_sw, greens_sw, misfit_sw, origin, grid)

    if comm.rank==0:
        print('Evaluating polarity and amplitude misfit...\n')

    #
    # The misfit function returns one scalar per trial source, so the
    # ordinary grid_search is used, exactly as for the waveform and
    # polarity misfits
    #

    observations = polarities_pamp_spamp_ratio_from_dict(
        polarities_dict, pamp_dict, spamp_dict, stations)

    results_polarity_amplitude = grid_search(
        observations, greens_bw, polarity_amplitude_misfit, origin, grid)


    if comm.rank==0:

        results = results_bw + results_sw

        # `grid` index corresponding to minimum misfit
        idx = results.source_idxmin()

        best_mt = grid.get(idx)
        lune_dict = grid.get_dict(idx)


        #
        # Generate figures and save results
        #

        print('Generating figures...\n')

        plot_data_greens2(event_id+'FMT_waveforms.png',
            data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw,
            misfit_bw, misfit_sw, stations, origin, best_mt, lune_dict)

        plot_beachball(event_id+'FMT_beachball_waveforms.png',
            best_mt, stations, origin)

        plot_misfit_lune(event_id+'FMT_misfit_waveforms.png', results,
            title='Waveform Misfit')

        # rendered the same way as the waveform misfit above, so that the
        # two landscapes can be compared directly.
        plot_misfit_lune(
            event_id+'FMT_misfit_polarity_amplitude.png',
            results_polarity_amplitude,
            title='P-Pol + P-Amp + S/P Amp Misfit')

        # predicted polarities
        predicted = polarity_amplitude_misfit.get_predicted(
            greens_bw, best_mt)[0]

        # station attributes
        attrs = polarity_amplitude_misfit.collect_attributes(
            observations, greens_bw)

        plot_polarities(
            event_id+'FMT_beachball_polarity_amplitude.png',
            observations[0], predicted, attrs, origin, best_mt)

        print('\nFinished\n')
