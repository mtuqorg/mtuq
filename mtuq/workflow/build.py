"""Construction of native MTUQ objects from normalized workflow recipes."""

import copy
from collections import namedtuple

import numpy as np

from mtuq.event import Origin
from mtuq.misfit import WaveformMisfit
from mtuq.process_data import ProcessData
from mtuq.util.cap import Trapezoid as CapTrapezoid
from mtuq.util.math import lat_lon_tuples
from mtuq.wavelet import (
    EarthquakeTrapezoid,
    Gabor,
    GaborWavelet,
    Gaussian,
    RickerWavelet,
    Trapezoid as WaveletTrapezoid,
    Triangle,
)

from .config import (
    WORKFLOW_VERSION,
    WorkflowConfigError,
    _SOURCE_FUNCTIONS,
    _signature_defaults,
)
from .io import _mtuq_provenance, _plain_value


PreparedWorkflow = namedtuple(
    'PreparedWorkflow',
    'catalog_origin origins grid wavelet wavelet_config processors misfits '
    'resolved',
)


def _construct_native(function, kwargs, path):
    try:
        return function(**kwargs)
    except (AssertionError, TypeError, ValueError) as exc:
        detail = str(exc).strip() or type(exc).__name__
        raise WorkflowConfigError('%s: %s' % (path, detail)) from exc


def _build_origin(config):
    event = config['event']
    try:
        return Origin({
            'time': event['time'],
            'latitude': event['latitude'],
            'longitude': event['longitude'],
            'depth_in_m': event['depth_in_m'],
        })
    except (AssertionError, TypeError, ValueError) as exc:
        detail = str(exc).strip() or type(exc).__name__
        raise WorkflowConfigError('event: %s' % detail) from exc


def _build_origins(config, catalog_origin):
    """Builds searched origins, or returns ``None`` for a fixed origin."""
    if 'origin_search' not in config:
        return None

    search = config['origin_search']
    if 'depth_in_m' in search:
        depths = search['depth_in_m']['values']
    else:
        depths = [catalog_origin.depth_in_m]

    if 'hypocenter' in search:
        hypocenter = search['hypocenter']
        locations = lat_lon_tuples(
            center_lat=catalog_origin.latitude,
            center_lon=catalog_origin.longitude,
            spacing_in_m=hypocenter['spacing_in_m'],
            npts_per_edge=hypocenter['npts_per_edge'],
        )
        locations = list(locations)
        reference_location = (
            catalog_origin.latitude,
            catalog_origin.longitude,
        )
        if not any(
            np.allclose(
                location,
                reference_location,
                rtol=0.0,
                atol=1.0e-10,
            )
            for location in locations
        ):
            locations.append(reference_location)
    else:
        locations = [
            (catalog_origin.latitude, catalog_origin.longitude),
        ]

    origins = []
    for depth in depths:
        for latitude, longitude in locations:
            origin = catalog_origin.copy()
            origin.latitude = latitude
            origin.longitude = longitude
            origin.depth_in_m = depth
            origins.append(origin)
    return origins


def _build_grid(config):
    source = config['source']
    grid_cfg = source['grid']
    function = _SOURCE_FUNCTIONS[(source['type'], grid_cfg['type'])]
    kwargs = {'magnitudes': list(source['magnitudes'])}
    if grid_cfg['type'] == 'regular':
        kwargs['npts_per_axis'] = int(grid_cfg['npts_per_axis'])
        if source['type'] in {'dev', 'fmt'}:
            kwargs['tightness'] = float(grid_cfg['tightness'])
            kwargs['uniformity'] = float(grid_cfg['uniformity'])
    else:
        kwargs['npts'] = int(grid_cfg['npts'])
    return _construct_native(function, kwargs, 'source.grid')


def _build_wavelet(config):
    wavelet_cfg = config.get('wavelet', {'type': 'trapezoid'})
    wavelet_type = wavelet_cfg['type']

    if wavelet_type == 'trapezoid':
        if 'rise_time' in wavelet_cfg:
            function = WaveletTrapezoid
            kwargs = {
                'rise_time': float(wavelet_cfg['rise_time']),
                'half_duration': float(wavelet_cfg['half_duration']),
            }
        else:
            function = CapTrapezoid
            if 'magnitude' in wavelet_cfg:
                magnitude = float(wavelet_cfg['magnitude'])
            else:
                magnitude = float(
                    np.median(
                        np.asarray(
                            config['source']['magnitudes'],
                            dtype=float,
                        )
                    )
                )
            kwargs = {'magnitude': magnitude}
    else:
        functions = {
            'triangle': (Triangle, ('half_duration',)),
            'gaussian': (Gaussian, ('sigma', 'mu')),
            'gabor': (Gabor, ('a', 'b')),
            'earthquake_trapezoid': (
                EarthquakeTrapezoid,
                ('rise_time', 'rupture_time'),
            ),
            'ricker': (RickerWavelet, ('dominant_frequency',)),
            'gabor_wavelet': (
                GaborWavelet,
                ('dominant_frequency',),
            ),
        }
        function, parameter_names = functions[wavelet_type]
        defaults = _signature_defaults(function)
        kwargs = {}
        for name in parameter_names:
            if name in wavelet_cfg:
                kwargs[name] = float(wavelet_cfg[name])
            elif defaults.get(name) is not None:
                kwargs[name] = float(defaults[name])

    wavelet = _construct_native(function, kwargs, 'wavelet')
    resolved = {
        'type': wavelet_type,
        'function': '%s.%s' % (function.__module__, function.__name__),
    }
    resolved.update(kwargs)
    return wavelet, resolved


def _build_measurements(config):
    processors = {}
    misfits = {}
    resolved_measurements = {}

    for name, measurement in config['measurements'].items():
        processing_kwargs = copy.deepcopy(measurement['processing'])
        processor = _construct_native(
            ProcessData, processing_kwargs, 'measurements.%s.processing' % name
        )

        waveform = measurement['waveform_misfit']
        misfit_kwargs = {
            'norm': config['misfit']['norm'],
            'normalize': config['misfit']['normalize'],
            'level': config['misfit']['level'],
            'verbose': config['misfit']['verbose'],
            'time_shift_min': waveform['time_shift_min'],
            'time_shift_max': waveform['time_shift_max'],
            'time_shift_groups': list(waveform['time_shift_groups']),
        }
        misfit = _construct_native(
            WaveformMisfit,
            misfit_kwargs,
            'measurements.%s.waveform_misfit' % name,
        )

        processors[name] = processor
        misfits[name] = misfit
        resolved_measurement = {
            'processing': _resolved_processing(processor, processing_kwargs),
            'waveform_misfit': _resolved_waveform_misfit(misfit),
        }
        if 'role' in measurement:
            resolved_measurement['role'] = measurement['role']
        resolved_measurements[name] = resolved_measurement

    return processors, misfits, resolved_measurements


def _resolved_processing(processor, supplied_kwargs):
    defaults = _signature_defaults(ProcessData.__init__)
    effective = dict(defaults)
    effective.update(supplied_kwargs)

    keys = [
        'filter_type',
        'freq',
        'freq_min',
        'freq_max',
        'pick_type',
        'taup_model',
        'FK_database',
        'FK_model',
        'window_type',
        'window_length',
        'apply_padding',
        'apply_statics',
        'time_shift_min',
        'time_shift_max',
        'apply_weights',
        'apply_scaling',
        'scaling_power',
        'scaling_coefficient',
        'capuaf_file',
    ]

    resolved = {}
    for key in keys:
        if (
            key in {'time_shift_min', 'time_shift_max'}
            and not processor.apply_padding
        ):
            continue
        if key in {'freq', 'freq_min', 'freq_max'} and hasattr(processor, key):
            value = getattr(processor, key)
        elif key in {'scaling_power', 'scaling_coefficient'}:
            if not processor.apply_scaling:
                continue
            if hasattr(processor, key):
                value = getattr(processor, key)
            else:
                value = effective.get(key)
        else:
            value = effective.get(key)
        if value is not None:
            resolved[key] = _plain_value(value)

    return resolved


def _resolved_waveform_misfit(misfit):
    return {
        'time_shift_min': float(misfit.time_shift_min),
        'time_shift_max': float(misfit.time_shift_max),
        'time_shift_groups': list(misfit.time_shift_groups),
    }


def _resolve_config(
    normalized, grid, resolved_wavelet, resolved_measurements
):
    """ Builds the complete configuration written to config.resolved.yaml

    Includes object-derived grid metadata, effective measurement settings,
    effective wavelet settings, objective coefficients, and provenance.
    """
    resolved = copy.deepcopy(normalized)

    source = resolved['source']
    grid_cfg = source['grid']
    grid_function = _SOURCE_FUNCTIONS[(source['type'], grid_cfg['type'])]
    grid_cfg['function'] = 'mtuq.grid.%s' % grid_function.__name__
    grid_cfg['size'] = int(grid.size)

    resolved['wavelet'] = copy.deepcopy(resolved_wavelet)

    resolved['measurements'] = resolved_measurements

    names = list(normalized['measurements'])
    if 'objective' in normalized:
        resolved['objective'] = copy.deepcopy(normalized['objective'])
    else:
        resolved['objective'] = {
            'coefficients': {name: 1.0 for name in names},
        }

    resolved['workflow'] = {'version': WORKFLOW_VERSION}
    resolved['mtuq'] = _mtuq_provenance()

    return resolved


def prepare_workflow(normalized):
    """Constructs native objects and the resolved configuration.

    This is the shared construction boundary used by execution, ``--validate``,
    and catalog validation.
    """
    catalog_origin = _build_origin(normalized)
    origins = _build_origins(normalized, catalog_origin)
    grid = _build_grid(normalized)
    wavelet, resolved_wavelet = _build_wavelet(normalized)
    processors, misfits, resolved_measurements = _build_measurements(
        normalized
    )
    resolved = _resolve_config(
        normalized,
        grid=grid,
        resolved_wavelet=resolved_wavelet,
        resolved_measurements=resolved_measurements,
    )
    return PreparedWorkflow(
        catalog_origin=catalog_origin,
        origins=origins,
        grid=grid,
        wavelet=wavelet,
        wavelet_config=resolved_wavelet,
        processors=processors,
        misfits=misfits,
        resolved=resolved,
    )
