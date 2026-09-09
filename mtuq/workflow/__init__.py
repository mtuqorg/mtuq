"""YAML-driven execution of ordinary MTUQ inversions.

The workflow layer translates an event recipe into the same native MTUQ
objects and operations used by an ordinary processing script.  Configuration
is loaded and normalized in :mod:`mtuq.workflow.config`; native ``Origin``,
grid, wavelet, ``ProcessData``, and ``WaveformMisfit`` objects are assembled in
:mod:`mtuq.workflow.build`; :func:`run` performs data and Green's-function I/O,
processing, grid search, result combination, and output writing.  Catalog
orchestration launches that same single-event path in isolated processes.

The stable public interface is ``run()``, ``validate_config()``,
``run_catalog()``, and ``WorkflowConfigError``.
"""

from .config import WorkflowConfigError as WorkflowConfigError
from .config import validate_config as validate_config
from .run import main as main
from .run import run as run


def run_catalog(
    directory, validate_only=False, resume=False, summary_only=False,
    timeout=None,
):
    """Runs or summarizes a directory of event recipes."""
    from .catalog import run_catalog as _run_catalog
    return _run_catalog(
        directory,
        validate_only=validate_only,
        resume=resume,
        summary_only=summary_only,
        timeout=timeout,
    )


__all__ = [
    'WorkflowConfigError',
    'main',
    'run',
    'run_catalog',
    'validate_config',
]
