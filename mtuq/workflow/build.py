"""Construction of native MTUQ objects from normalized workflow recipes."""

import copy
from collections import namedtuple

import numpy as np

from mtuq.event import Origin
from mtuq.misfit import WaveformMisfit
from mtuq.process_data import ProcessData
from mtuq.util.cap import Trapezoid

from .config import (
    WORKFLOW_VERSION,
    WorkflowConfigError,
    _SOURCE_FUNCTIONS,
    _signature_defaults,
)
from .io import _mtuq_provenance, _plain_value


PreparedWorkflow = namedtuple(
    'PreparedWorkflow',
    'origin grid wavelet wavelet_magnitude processors misfits resolved',
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
    if 'magnitude' in wavelet_cfg:
        magnitude = float(wavelet_cfg['magnitude'])
    else:
        magnitude = float(
            np.median(np.asarray(config['source']['magnitudes'], dtype=float))
        )
    wavelet = _construct_native(
        Trapezoid, {'magnitude': magnitude}, 'wavelet'
    )
    return wavelet, magnitude


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
        resolved_measurements[name] = {
            'processing': _resolved_processing(processor, processing_kwargs),
            'waveform_misfit': _resolved_waveform_misfit(misfit),
        }

    return processors, misfits, resolved_measurements


def _resolved_processing(processor, supplied_kwargs):
    defaults = _signature_defaults(ProcessData.__init__)
    effective = dict(defaults)
    effective.update(supplied_kwargs)

    keys = [
        'filter_type',
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
        if key in {'freq_min', 'freq_max'} and hasattr(processor, key):
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
    normalized, grid, wavelet_magnitude, resolved_measurements
):
    """ Builds the complete configuration written to config.resolved.yaml

    Includes object-derived grid metadata, effective measurement settings,
    wavelet magnitude, objective coefficients, and provenance.
    """
    resolved = copy.deepcopy(normalized)

    source = resolved['source']
    grid_cfg = source['grid']
    grid_function = _SOURCE_FUNCTIONS[(source['type'], grid_cfg['type'])]
    grid_cfg['function'] = 'mtuq.grid.%s' % grid_function.__name__
    grid_cfg['size'] = int(grid.size)

    wavelet = resolved.setdefault('wavelet', {'type': 'trapezoid'})
    wavelet['type'] = 'trapezoid'
    wavelet['function'] = 'mtuq.util.cap.Trapezoid'
    wavelet['magnitude'] = float(wavelet_magnitude)

    resolved['measurements'] = resolved_measurements

    names = list(normalized['measurements'])
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
    origin = _build_origin(normalized)
    grid = _build_grid(normalized)
    wavelet, wavelet_magnitude = _build_wavelet(normalized)
    processors, misfits, resolved_measurements = _build_measurements(
        normalized
    )
    resolved = _resolve_config(
        normalized,
        grid=grid,
        wavelet_magnitude=wavelet_magnitude,
        resolved_measurements=resolved_measurements,
    )
    return PreparedWorkflow(
        origin=origin,
        grid=grid,
        wavelet=wavelet,
        wavelet_magnitude=wavelet_magnitude,
        processors=processors,
        misfits=misfits,
        resolved=resolved,
    )
