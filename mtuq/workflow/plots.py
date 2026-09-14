"""Standard plots for completed MTUQ workflow inversions."""

from pathlib import Path

from mtuq.graphics import (
    plot_beachball,
    plot_data_greens1,
    plot_data_greens2,
    plot_data_greens3,
    plot_misfit_dc,
    plot_misfit_depth,
    plot_misfit_latlon,
    plot_misfit_lune,
    plot_misfit_vw,
)


_SOURCE_MISFIT_PLOTS = {
    'dc': plot_misfit_dc,
    'dev': plot_misfit_vw,
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

    search = config.get('origin_search')
    if search is None:
        return

    if config['source']['grid']['type'] != 'regular':
        _print_skip(
            'origin misfit',
            'native origin-search plots require a regular source grid',
        )
        return

    if set(search) == {'depth_in_m', 'hypocenter'}:
        _print_skip(
            'origin misfit',
            'combined depth and hypocenter searches have no native 2-D plot',
        )
        return

    if 'depth_in_m' in search:
        _call_plot(
            'misfit_depth',
            plot_misfit_depth,
            _plot_filename(plot_dir, event_id, 'misfit_depth'),
            results,
            origins,
        )
    elif 'hypocenter' in search:
        _call_plot(
            'misfit_latlon',
            plot_misfit_latlon,
            _plot_filename(plot_dir, event_id, 'misfit_latlon'),
            results,
            origins,
        )


def _plot_filename(plot_dir, event_id, plot_type):
    return Path(plot_dir) / ('%s_%s.png' % (event_id, plot_type))


def _call_plot(label, function, filename, *args):
    try:
        function(str(filename), *args)
    except Exception as exc:
        detail = str(exc).strip() or type(exc).__name__
        print("  Plot %r failed: %s" % (label, detail))


def _print_skip(label, reason):
    print("  Skipping plot %r: %s" % (label, reason))
