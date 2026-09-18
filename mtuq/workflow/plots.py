"""Standard plots for completed MTUQ workflow inversions."""

from pathlib import Path

import numpy as np

from mtuq.graphics import (
    plot_beachball,
    plot_confidence_curve,
    plot_data_greens1,
    plot_data_greens2,
    plot_data_greens3,
    plot_misfit_dc,
    plot_misfit_depth,
    plot_misfit_latlon,
    plot_misfit_lune,
)
from mtuq.misfit.waveform import calculate_norm_data, estimate_sigma


_SOURCE_MISFIT_PLOTS = {
    'dc': plot_misfit_dc,
    'dev': plot_misfit_lune,
    'fmt': plot_misfit_lune,
}


def generate_plots(
    config,
    output_dir,
    results,
    data,
    greens,
    processors,
    misfits,
    stations,
    origin,
    source,
    source_dict,
    origins=None,
    term_results=None,
    origin_idx=0,
):
    """Generates requested public-MTUQ plots after results are saved."""
    plot_dir = Path(output_dir) / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    event_id = config['event']['id']

    for name in config['plots']:
        if name == 'waveform':
            _plot_waveform(
                plot_dir,
                event_id,
                config['measurements'],
                data,
                greens,
                processors,
                misfits,
                stations,
                origin,
                source,
                source_dict,
            )
        elif name == 'beachball':
            _call_plot(
                'beachball',
                plot_beachball,
                _plot_filename(plot_dir, event_id, 'beachball'),
                source,
                stations,
                origin,
            )
        elif name == 'misfit':
            _plot_misfit(
                plot_dir,
                event_id,
                config,
                results,
                origins,
            )
        elif name == 'confidence':
            try:
                _plot_confidence(
                    plot_dir,
                    event_id,
                    config,
                    term_results,
                    data,
                    greens,
                    misfits,
                    origin,
                    origin_idx,
                    source,
                    source_dict,
                )
            except Exception as exc:
                detail = str(exc).strip() or type(exc).__name__
                print("  Plot 'confidence' failed: %s" % detail)


def _plot_waveform(
    plot_dir,
    event_id,
    measurements,
    data,
    greens,
    processors,
    misfits,
    stations,
    origin,
    source,
    source_dict,
):
    names = list(measurements)

    if len(names) == 1:
        name = names[0]
        _plot_waveform_term(
            plot_dir, event_id, 'waveform', name,
            data, greens, processors, misfits,
            stations, origin, source, source_dict,
        )
        return

    roles = {
        measurement['role']: name
        for name, measurement in measurements.items()
        if 'role' in measurement
    }

    surface_role = next(
        (
            role
            for role in ('surface', 'rayleigh', 'love')
            if role in roles
        ),
        None,
    )
    if (
        len(names) == 2
        and surface_role is not None
        and set(roles) == {'body', surface_role}
    ):
        body = roles['body']
        surface = roles[surface_role]
        _call_plot(
            'waveform',
            plot_data_greens2,
            _plot_filename(plot_dir, event_id, 'waveform'),
            data[body],
            data[surface],
            greens[body],
            greens[surface],
            processors[body],
            processors[surface],
            misfits[body],
            misfits[surface],
            stations,
            origin,
            source,
            source_dict,
        )
        return

    if len(names) == 3 and set(roles) == {'body', 'rayleigh', 'love'}:
        body = roles['body']
        rayleigh = roles['rayleigh']
        love = roles['love']
        _call_plot(
            'waveform',
            plot_data_greens3,
            _plot_filename(plot_dir, event_id, 'waveform'),
            data[body],
            data[rayleigh],
            data[love],
            greens[body],
            greens[rayleigh],
            greens[love],
            processors[body],
            processors[rayleigh],
            processors[love],
            misfits[body],
            misfits[rayleigh],
            misfits[love],
            stations,
            origin,
            source,
            source_dict,
        )
        return

    for name in names:
        _plot_waveform_term(
            plot_dir, event_id, 'waveform_%s' % name, name,
            data, greens, processors, misfits,
            stations, origin, source, source_dict,
        )


def _plot_waveform_term(
    plot_dir,
    event_id,
    plot_type,
    name,
    data,
    greens,
    processors,
    misfits,
    stations,
    origin,
    source,
    source_dict,
):
    _call_plot(
        plot_type,
        plot_data_greens1,
        _plot_filename(plot_dir, event_id, plot_type),
        data[name],
        greens[name],
        processors[name],
        misfits[name],
        stations,
        origin,
        source,
        source_dict,
    )


def _plot_misfit(plot_dir, event_id, config, results, origins):
    source_type = config['source']['type']
    _call_plot(
        'misfit',
        _SOURCE_MISFIT_PLOTS[source_type],
        _plot_filename(plot_dir, event_id, 'misfit'),
        results,
    )
    if source_type in {'dev', 'fmt'}:
        _call_plot(
            'misfit_dc',
            plot_misfit_dc,
            _plot_filename(plot_dir, event_id, 'misfit_dc'),
            results,
        )

    search = config.get('origin_search')
    if search is None:
        return

    if config['source']['grid']['type'] != 'regular':
        _print_skip(
            'origin misfit',
            'native origin-search plots require a regular source grid',
        )
        return

    depths = {origin.depth_in_m for origin in origins}
    locations = {
        (origin.latitude, origin.longitude) for origin in origins
    }
    varies_in_depth = len(depths) > 1
    varies_horizontally = len(locations) > 1

    if varies_in_depth and varies_horizontally:
        _print_skip(
            'origin misfit',
            'combined depth and hypocenter searches have no native 2-D plot',
        )
        return

    if varies_in_depth:
        _call_plot(
            'misfit_depth',
            plot_misfit_depth,
            _plot_filename(plot_dir, event_id, 'misfit_depth'),
            results,
            origins,
            show_tradeoffs=True,
            show_magnitudes=True,
            title=event_id,
        )
    elif varies_horizontally:
        _call_plot(
            'misfit_latlon',
            plot_misfit_latlon,
            _plot_filename(plot_dir, event_id, 'misfit_latlon'),
            results,
            origins,
            show_tradeoffs=True,
        )
    else:
        _print_skip(
            'origin misfit',
            'searched origins do not vary in depth or hypocenter',
        )


def _plot_confidence(
    plot_dir,
    event_id,
    config,
    term_results,
    data,
    greens,
    misfits,
    origin,
    origin_idx,
    source,
    source_dict,
):
    """Plots one confidence curve from the joint measurement likelihood.

    Each measurement surface is conditioned on the reported origin and scalar
    moment, then divided by its own public MTUQ a posteriori variance estimate.
    Summing those dimensionless surfaces is equivalent to multiplying the
    independent per-measurement likelihoods.
    """
    if term_results is None:
        raise ValueError('measurement result surfaces are unavailable')

    rho = source_dict['rho']
    combined = None

    for name in config['measurements']:
        misfit = misfits[name]
        components = _measurement_components(misfit)
        selected_greens = greens[name].select(origin)
        sigma = estimate_sigma(
            data[name],
            selected_greens,
            source,
            misfit.norm,
            components,
            misfit.time_shift_min,
            misfit.time_shift_max,
        )
        variance = float(sigma)**2

        if misfit.normalize:
            data_norm = calculate_norm_data(
                data[name], misfit.norm, components
            )
            if not np.isfinite(data_norm) or data_norm <= 0.0:
                raise ValueError(
                    "measurement %r has invalid data norm %r"
                    % (name, data_norm)
                )
            variance /= data_norm

        if not np.isfinite(variance) or variance <= 0.0:
            raise ValueError(
                "measurement %r has invalid estimated variance %r"
                % (name, variance)
            )

        conditioned = _condition_confidence_results(
            term_results[name], origin_idx, rho
        )
        scaled = conditioned / variance
        combined = scaled if combined is None else combined + scaled

    if combined is None:
        raise ValueError('confidence requires at least one measurement')

    _call_plot(
        'confidence',
        plot_confidence_curve,
        _plot_filename(plot_dir, event_id, 'confidence'),
        combined,
        1.0,
        m0=source,
        normalized=False,
    )

def _measurement_components(misfit):
    components = []
    for group in misfit.time_shift_groups:
        for component in group:
            if component not in components:
                components.append(component)
    return components


def _condition_confidence_results(results, origin_idx, rho):
    names = set(results.index.names)
    missing = {'origin_idx', 'rho'} - names
    if missing:
        raise ValueError(
            'confidence result index is missing %s'
            % ', '.join(sorted(missing))
        )

    origins = results.index.get_level_values('origin_idx')
    magnitudes = results.index.get_level_values('rho')
    mask = (origins == origin_idx) & np.isclose(
        magnitudes, rho, rtol=1.e-10, atol=0.0
    )
    conditioned = results.loc[mask].copy()
    if conditioned.empty:
        raise ValueError(
            'no confidence samples match origin_idx=%r and rho=%r'
            % (origin_idx, rho)
        )
    return conditioned


def _plot_filename(plot_dir, event_id, plot_type):
    return Path(plot_dir) / ('%s_%s.png' % (event_id, plot_type))


def _call_plot(label, function, filename, *args, **kwargs):
    try:
        function(str(filename), *args, **kwargs)
    except Exception as exc:
        detail = str(exc).strip() or type(exc).__name__
        print("  Plot %r failed: %s" % (label, detail))


def _print_skip(label, reason):
    print("  Skipping plot %r: %s" % (label, reason))
