"""Directory orchestration for MTUQ workflow recipes."""

import argparse
import csv
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from .build import prepare_workflow
from .config import WorkflowConfigError, validate_config


_SUMMARY_FIELDS = [
    'event_id', 'status', 'Mw', 'latitude', 'longitude', 'depth_in_m',
    'recipe', 'output',
]


def _discover_recipes(directory):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise WorkflowConfigError(
            'catalog path must be a directory: %s' % directory
        )

    recipes = sorted(
        set(directory.glob('*.yaml')) | set(directory.glob('*.yml')),
        key=lambda path: path.name,
    )
    if not recipes:
        raise WorkflowConfigError(
            'no YAML event recipes found in %s' % directory
        )
    return directory, recipes


def _validate_recipe(path):
    """ Validates a recipe and constructs its MTUQ objects
    """
    config = validate_config(path)
    prepare_workflow(config)
    return {
        'path': Path(path).resolve(),
        'event_id': str(config['event']['id']),
        'output': Path(config['output']),
        'config': config,
    }


def _validate_recipes(paths):
    recipes, errors = [], []
    for path in paths:
        try:
            recipes.append(_validate_recipe(path))
        except WorkflowConfigError as exc:
            errors.append((path, str(exc)))

    unique, event_ids, outputs = [], {}, {}
    for recipe in recipes:
        event_id = recipe['event_id']
        if event_id in event_ids:
            errors.append((
                recipe['path'],
                'event id %r also appears in %s'
                % (event_id, event_ids[event_id].name),
            ))
            continue

        output = os.path.normcase(str(recipe['output'].resolve()))
        if output in outputs:
            errors.append((
                recipe['path'],
                'output directory %s is also used by %s'
                % (recipe['output'], outputs[output].name),
            ))
            continue

        event_ids[event_id] = recipe['path']
        outputs[output] = recipe['path']
        unique.append(recipe)

    return unique, errors


def _print_errors(errors):
    for path, message in errors:
        print(
            '%s: cannot be used as a catalog event recipe\n  %s'
            % (path.name, message)
        )
        print('  If this file is not an event recipe, move it out of the '
              'catalog directory.')


def _result_total(output):
    for extension in ('.nc', '.h5'):
        path = Path(output) / ('results_total' + extension)
        if path.is_file():
            return path
    return None


def _is_complete(recipe):
    output = recipe['output']
    solution_path = output / 'solution.json'
    if not (
        (output / 'config.resolved.yaml').is_file()
        and solution_path.is_file()
        and _result_total(output) is not None
    ):
        return False

    try:
        with solution_path.open('r', encoding='utf-8') as handle:
            solution = json.load(handle)
    except (OSError, ValueError):
        return False
    return isinstance(solution, Mapping)


def _run_event_process(path, timeout=None):
    return subprocess.run(
        [sys.executable, '-m', 'mtuq.workflow', str(path)], timeout=timeout
    )


def _compact_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return format(value, '.15g')


def _summary_row(recipe):
    event = recipe['config']['event']
    row = {
        'event_id': recipe['event_id'],
        'status': 'complete' if _is_complete(recipe) else 'incomplete',
        'Mw': None,
        'latitude': event['latitude'],
        'longitude': event['longitude'],
        'depth_in_m': event['depth_in_m'],
        'recipe': str(recipe['path']),
        'output': str(recipe['output']),
    }
    if row['status'] == 'complete':
        try:
            with (recipe['output'] / 'solution.json').open(
                'r', encoding='utf-8'
            ) as f:
                solution = json.load(f)
        except (OSError, ValueError):
            row['status'] = 'incomplete'
        else:
            if not isinstance(solution, Mapping):
                row['status'] = 'incomplete'
                solution = {}
            row['Mw'] = solution.get('Mw')
            for key in ('latitude', 'longitude', 'depth_in_m'):
                row[key] = solution.get(key, row[key])

    for key in ('Mw', 'latitude', 'longitude', 'depth_in_m'):
        row[key] = _compact_number(row[key])
    return row


def _write_summary(directory, recipes, errors=()):
    rows = [_summary_row(recipe) for recipe in recipes]
    rows.extend({
        'event_id': '', 'status': 'invalid', 'Mw': '', 'latitude': '',
        'longitude': '', 'depth_in_m': '', 'recipe': str(Path(path).resolve()),
        'output': '',
    } for path, _ in errors)
    rows.sort(key=lambda row: Path(row['recipe']).name)

    path = Path(directory) / 'catalog_summary.csv'
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _mpi_world_size():
    try:
        from mpi4py import MPI
    except ImportError:
        return 1
    return MPI.COMM_WORLD.Get_size()


def run_catalog(
    directory, validate_only=False, resume=False, summary_only=False,
    timeout=None,
):
    """ Runs or summarizes MTUQ event recipes from a directory

    .. rubric :: Input arguments

    ``directory`` (`str` or `os.PathLike`)
        Directory containing complete single-event YAML recipes.

    ``validate_only`` (`bool`)
        Validate recipes without running inversions.

    ``resume`` (`bool`)
        Skip recipes whose output directories are already complete.

    ``summary_only`` (`bool`)
        Write ``catalog_summary.csv`` from existing outputs without running.

    ``timeout`` (`float`)
        Optional maximum runtime in seconds for each event.

    .. rubric :: Returns

    Dictionary containing the validated recipes, failures, and summary path.
    """
    if _mpi_world_size() > 1:
        raise WorkflowConfigError(
            'mtuq-catalog must not be launched with MPI; run each event YAML '
            'as its own scheduler/MPI job instead'
        )

    directory, paths = _discover_recipes(directory)
    recipes, errors = _validate_recipes(paths)

    if summary_only:
        if errors:
            _print_errors(errors)
        summary = _write_summary(directory, recipes, errors)
        print('Wrote %s' % summary)
        if errors:
            print('%d invalid recipe(s) included in summary' % len(errors))
        return {
            'valid': not errors, 'recipes': recipes,
            'failures': errors, 'summary': summary,
        }

    if errors:
        _print_errors(errors)
        print('\n%d invalid recipe(s); catalog not started' % len(errors))
        return {'valid': False, 'recipes': recipes, 'failures': errors}

    if validate_only:
        for recipe in recipes:
            print('%s: valid' % recipe['path'].name)
        print('\n%d recipe(s) valid' % len(recipes))
        return {'valid': True, 'recipes': recipes, 'failures': []}

    failures = []
    for index, recipe in enumerate(recipes, start=1):
        label = '[%d/%d] %s' % (index, len(recipes), recipe['path'].name)
        if resume and _is_complete(recipe):
            print('%s: already complete; skipping' % label)
            continue

        print('%s: running' % label)
        try:
            completed = _run_event_process(recipe['path'], timeout=timeout)
        except subprocess.TimeoutExpired:
            message = 'timed out after %g s' % timeout
            failures.append((recipe['path'], message))
            print('%s: failed: %s' % (label, message))
            continue
        except OSError as exc:
            failures.append((recipe['path'], str(exc)))
            print('%s: failed: %s' % (label, exc))
            continue

        if completed.returncode != 0:
            message = 'mtuq-run exited with status %d' % completed.returncode
            failures.append((recipe['path'], message))
            print('%s: failed: %s' % (label, message))
        elif not _is_complete(recipe):
            message = 'mtuq-run finished but expected output is incomplete'
            failures.append((recipe['path'], message))
            print('%s: failed: %s' % (label, message))
        else:
            print('%s: complete' % label)

    summary = _write_summary(directory, recipes)
    print('\nWrote %s' % summary)
    if failures:
        names = ', '.join(path.name for path, _ in failures)
        print('%d of %d event(s) failed: %s'
              % (len(failures), len(recipes), names))
    else:
        print('All %d event(s) complete' % len(recipes))
    return {
        'valid': True,
        'recipes': recipes,
        'failures': failures,
        'summary': summary,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='mtuq-catalog',
        description='Run all MTUQ event YAML recipes in a directory.',
    )
    parser.add_argument(
        'directory', help='directory containing event YAML files'
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--validate', action='store_true',
                      help='validate all recipes without running them')
    mode.add_argument('--resume', action='store_true',
                      help='skip events that already have complete outputs')
    mode.add_argument('--summary', action='store_true',
                      help='summarize existing event outputs')
    parser.add_argument('--timeout', type=float, metavar='SECONDS',
                        help='optional maximum runtime for each event')
    args = parser.parse_args(argv)

    if args.timeout is not None and args.timeout <= 0:
        parser.error('--timeout must be positive')

    result = run_catalog(
        args.directory,
        validate_only=args.validate,
        resume=args.resume,
        summary_only=args.summary,
        timeout=args.timeout,
    )
    return 0 if result['valid'] and not result['failures'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
