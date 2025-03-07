
Moment tensor and force grids
=============================


Currently implemented moment tensor grids
-----------------------------------------

- `Randomly-drawn full moment tensors <https://mtuqorg.github.io/mtuq/library/generated/mtuq.grid.FullMomentTensorGridRandom.html>`_
- `Regularly-spaced full moment tensors <https://mtuqorg.github.io/mtuq/library/generated/mtuq.grid.FullMomentTensorGridSemiregular.html>`_
- `Randomly-drawn double couples <https://mtuqorg.github.io/mtuq/library/generated/mtuq.grid.DoubleCoupleGridRandom.html>`_
- `Regularly-spaced double couples <https://mtuqorg.github.io/mtuq/library/generated/mtuq.grid.DoubleCoupleGridRegular.html>`_
- `Randomly-drawn deviatoric moment tensors <https://mtuqorg.github.io/mtuq/library/generated/mtuq.grid.DeviatoricGridRandom.html>`_
- `Regularly-spaced deviatoric moment tensors <https://mtuqorg.github.io/mtuq/library/generated/mtuq.grid.DeviatoricGridSemiregular.html>`_


Currently implemented force grids
---------------------------------

- `Randomly-drawn forces <https://mtuqorg.github.io/mtuq/library/generated/mtuq.grid.ForceGridRandom.html>`_
- `Regularly-spaced forces <https://mtuqorg.github.io/mtuq/library/generated/mtuq.grid.ForceGridRegular.html>`_



Using moment tensor grids
-------------------------

Moment tensor grids created in the above manner can be iterated over like a NumPy array or list, or individual grid points can be accessed by the following methods:

- Use ``get(i)`` to return the `i`-th moment tensor as a `MomentTensor` object
- Use ``get(i).as_vector()`` to return the `i`-th moment tensor as a NumPy array `(Mrr, Mtt, Mpp, Mrp, Mrt, Mtp)`
- Use ``get_dict(i)`` to return the `i`-th moment tensor as dictionary of Tape2015 parameters `rho, v, w, kappa, sigma, h`


Using force grids
-----------------

Force grids created in the above manner can be iterated over like a NumPy array or list, or individual grid points can be accessed by the following methods:

- Use ``get(i)`` to return the `i`-th force as a `Force` object
- Use ``get(i).as_vector()`` to return the `i`-th force as a NumPy array `(Fr, Ft, Fp)`
- Use ``get_dict(i)`` to return the `i`-th force as dictionary of parameters `F0, phi, h`


More details
------------

For additional usage information, see `Grid <https://mtuqorg.github.io/mtuq/library/generated/mtuq.grid.Grid.html>`_ for regularly-spaced grids or `UnstructuredGrid <https://mtuqorg.github.io/mtuq/library/generated/mtuq.grid.UnstructuredGrid.html>`_ for randomly-drawn grids.

Moment tensor grids are implemented using the `rho, v, w, kappa, sigma, h` parameterization from `Tape2015 <https://mtuqorg.github.io/mtuq/references.html>`_.  

Force grids are implemented using parameters `F0, phi, h`, which are related to `r, phi, theta` spherical coordinates (physics convention) by `F0 = r`, `phi = phi`, `h = cos(theta)`.  In addition, `F0, phi, h` are related to geographic directions by these `formulas <https://mtuqorg.github.io/mtuq/_modules/mtuq/util/math.html#to_rtp>`_.

Random moment tensors are drawn from the uniform distributions described by `Tape2015 <https://mtuqorg.github.io/mtuq/references.html>`_.


