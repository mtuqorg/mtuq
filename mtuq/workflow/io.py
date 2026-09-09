"""Input/output and provenance helpers for MTUQ workflows."""

import os
import shutil
import subprocess
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml

from mtuq import open_db
from mtuq.grid_search import MTUQDataArray, MTUQDataFrame


def _open_greens_database(greens):
    kwargs = {}
    if greens.get('model') is not None:
        kwargs['model'] = greens['model']
    if greens.get('cache_path') is not None:
        kwargs['cache_path'] = greens['cache_path']

    return open_db(
        path_or_url=greens.get('path', ''),
        format=greens['format'],
        **kwargs,
    )


def _save_native_results(output_dir, total_results, term_results):
    total_results.save(
        output_dir / ('results_total' + _result_extension(total_results))
    )
    for name, results in term_results.items():
        results.save(
            output_dir / ('results_%s%s' % (name, _result_extension(results)))
        )


def _result_extension(results):
    if isinstance(results, MTUQDataArray):
        return '.nc'
    if isinstance(results, MTUQDataFrame):
        return '.h5'
    raise TypeError(
        'unsupported native MTUQ result type: %s' % type(results).__name__
    )


def _write_input_config(recipe_path, output_dir):
    destination = output_dir / 'config.input.yaml'
    source = Path(recipe_path).expanduser().resolve()
    resolved_source = (output_dir / 'config.resolved.yaml').resolve()

    # Rerunning config.resolved.yaml in its own output directory must not
    # destroy the original abridged recipe already saved there.
    if source == resolved_source and destination.exists():
        return

    if source != destination.resolve():
        shutil.copyfile(source, destination)


def _write_yaml(path, data):
    with Path(path).open('w', encoding='utf-8') as handle:
        yaml.safe_dump(
            _plain_value(data),
            handle,
            sort_keys=False,
            default_flow_style=False,
        )


def _plain_value(value):
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value


def _resolve_path(path, base_dir):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path(base_dir) / path
    return str(path.resolve())


def _mtuq_provenance():
    try:
        version = metadata.version('mtuq')
    except metadata.PackageNotFoundError:
        version = None

    return {
        'version': version,
        'git_commit': _git_commit(),
    }


def _git_commit():
    package_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            [
                'git',
                '-C',
                str(package_root),
                'rev-parse',
                '--show-toplevel',
                'HEAD',
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    lines = result.stdout.splitlines()
    if len(lines) != 2:
        return None
    if Path(lines[0]).resolve() != package_root:
        return None

    commit = lines[1].strip()
    return commit or None


def _print_grid_summary(resolved):
    grid = resolved['source']['grid']
    print('Source grid')
    print('  function: %s' % grid['function'])
    print('  total grid points: {:,}'.format(grid['size']))
    print('')
