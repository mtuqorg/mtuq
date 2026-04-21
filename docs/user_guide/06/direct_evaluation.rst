Direct source evaluation
========================

Instead of searching over a grid, it is sometimes useful to evaluate a single, fixed moment tensor — for example, to forward-model waveforms, verify a published solution, or inspect fits before running a full inversion.

The example `examples/DirectEvaluation.py <https://github.com/mtuqorg/mtuq/blob/master/examples/DirectEvaluation.py>`_ demonstrates this workflow.  The source is defined directly from its moment tensor components, and ``from_mij`` converts them to the lune parameters needed by the plotting functions:

.. code::

    from mtuq.event import MomentTensor
    from mtuq.util.math import from_mij

    mt_dict = {
        'Mrr': -5800435174432632.0,
        'Mtt':   789342714264084.5,
        'Mpp':  5011092460168549.0,
        'Mrt':  3034156721905327.0,
        'Mrp':  2110077787951300.2,
        'Mtp':  2602039428905879.0,
        }

    mt_array  = list(mt_dict.values())
    lune_dict = dict(zip(('rho', 'v', 'w', 'kappa', 'sigma', 'h'), from_mij(mt_array)))
    mt        = MomentTensor(mt_array)

    plot_data_greens2(event_id + '_waveforms.png',
        data_bw, data_sw, greens_bw, greens_sw,
        process_bw, process_sw, misfit_bw, misfit_sw,
        stations, origin, mt, lune_dict)

    plot_beachball(event_id + '_beachball.png', mt, stations, origin)

If the source is known in terms of fault geometry rather than Mij components, ``to_mij`` and ``to_rho`` can be used to convert:

.. code::

    from mtuq.util.math import to_mij, to_rho

    Mw, strike, dip, rake = 4.5, 230., 40., -10.
    mt_array = to_mij(to_rho(Mw), 0., 0., strike, rake, np.cos(np.deg2rad(dip)))

Setting ``v = 0`` and ``w = 0`` in the ``to_mij`` call constrains the source to be a pure double couple.  For a full moment tensor, supply non-zero lune coordinates instead (see `Moment tensor and force grids <https://mtuqorg.github.io/mtuq/user_guide/06/moment_tensor_and_force_grids.html>`_) or use the `from_mij` function to convert Mij components to lune parameters.

You can find relevant utility functions in the ``mtuq.util.math`` module.