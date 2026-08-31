
import numpy as np

import mtuq
from mtuq.util import AttribDict, Null, warn
from mtuq.misfit.polarity import (
    _check, _get_azimuths, _radiation_coef_P, _takeoff_angles_taup)
from mtuq.misfit.waveform.level2 import _to_array, _type
from obspy.taup import TauPyModel


class PolarityPampSPampRatioMisfit(object):
    """ Polarity, P-amplitude and S/P-amplitude-ratio misfit function

    Evaluates the consistency of three body-wave observables with a trial
    moment tensor following Yang and Wang (2025, *Seismol. Res. Lett.*,
    96(2A), 1150-1162).


    .. rubric:: Observables

    Three quantities are compared. Each term can be enabled or disabled
    through its corresponding weight.

    1. **P-wave first-motion polarity**

       ``+1`` for positive/upward first motion, ``-1`` for negative/downward
       first motion, and ``0`` for unpicked or indeterminate observations.

    2. **P-wave amplitude**

       A signed scalar amplitude for each station. The amplitude magnitude
       is obtained from the maximum amplitude envelope within a short window
       beginning at the theoretical direct-P arrival. The sign of the vertical
       component at the sample corresponding to the envelope maximum determines
       the polarity and the sign of the P amplitude.

    3. **S/P amplitude ratio**

       A positive, linear S/P amplitude ratio derived from three-component,
       noise-corrected energies following Wang et al. (2020) and Yang and
       Wang (2025):

       .. code::

           rat = sqrt(
               [(S_R^2 + S_T^2 + S_Z^2)
                - (N_R^2 + N_T^2 + N_Z^2)]
               /
               [(P_R^2 + P_T^2 + P_Z^2)
                - (N_R^2 + N_T^2 + N_Z^2)]
           )

       where ``P``, ``S``, and ``N`` denote measurements from the P-wave,
       S-wave, and noise windows on the radial, transverse, and vertical
       components.

       The P and S windows begin at their corresponding theoretical direct
       arrivals, with window lengths based on a fraction of the S-P time.
       The noise window is chosen with the same duration before the event.

       This MTUQ implementation expects the observed S/P ratios as positive
       linear ratios and converts them internally to ``log10(S/P)`` when
       evaluating the misfit. Measurements recorded as logarithms are
       converted by passing ``spamp_ratio_is_log10=True`` to
       ``polarities_pamp_spamp_ratio_from_dict``. 

       Measurement of these observables is a preprocessing step outside this
       misfit class. They can be supplied using
       ``polarities_pamp_spamp_ratio_from_dict`` or equivalent arrays.


    .. rubric:: Predictions

    The predicted amplitudes are based on far-field body-wave radiation in a
    homogeneous isotropic whole-space approximation.

    The medium is one density and one pair of wave speeds for the whole
    propagation. By default these are read from the velocity model at the
    source depth, so that the medium and the takeoff angles come from the
    same model; see ``rho``, ``vp`` and ``vs`` below.

    Let ``R_P``, ``R_SV``, and ``R_SH`` denote the P, SV, and SH radiation
    coefficients and ``r`` the hypocentral distance. The predicted quantities
    are evaluated as

    .. code::

        amp_theo = R_P / (4 * pi * rho * vp**3 * r)

        log10_rat_theo = (
            log10(sqrt(R_SV**2 + R_SH**2) / abs(R_P))
            + 3 * log10(vp / vs)
        )

    ``R_P`` is evaluated using the direct-P takeoff angle. In this
    implementation, ``R_SV`` and ``R_SH`` are evaluated using the direct-S
    takeoff angle.

    The predicted P amplitude is evaluated using consistent units for scalar
    moment, density, seismic velocity, and hypocentral distance.

    .. rubric:: Objective function

    Following NaNDC (Yang and Wang, 2025) equations 3-6, the total objective is the weighted sum of
    three normalized L1 residual terms:

    .. code::

        Phi = (
            lambda_pol   * Phi_pol
            + lambda_pamp  * Phi_pamp
            + lambda_sprat * Phi_sprat
        )

        Phi_pol = (
            sum(0.5 * tau * abs(pol_obs - pol_theo))
            / sum(tau * abs(pol_obs))
        )

        Phi_pamp = (
            sum(tau * abs(amp_obs - amp_theo))
            / sum(tau * abs(amp_obs))
        )

        Phi_sprat = (
            sum(tau * abs(log10(rat_obs) - log10_rat_theo))
            / sum(tau * abs(log10(rat_obs)))
        )

    where ``tau`` represents the observation-validity or station-selection
    mask used for each term.

    Each component is normalized by the corresponding observed-data scale,
    making the three terms dimensionless before weighting.

    When combining this objective with other MTUQ misfit functions whose
    numerical scales differ substantially, appropriate relative weighting or
    rescaling should be considered.

    .. rubric:: Usage

    The misfit returns one scalar value for each trial source and can therefore
    be used with the standard MTUQ ``grid_search`` workflow:

    .. code::

        misfit = PolarityPampSPampRatioMisfit(
            taup_model='ak135'
        )

        observations = polarities_pamp_spamp_ratio_from_dict(
            polarities_dict,
            pamp_dict,
            spamp_dict,
            stations,
        )

        results = grid_search(
            observations,
            greens,
            misfit,
            origin,
            sources,
        )

    """

    def __init__(self,
        method='taup',
        taup_model='ak135',
        lambda_pol=1.,
        lambda_pamp=1.,
        lambda_sprat=1.,
        rho=None,
        vp=None,
        vs=None,
        norm='L1',
        radiation_floor=1.e-6):

        """
        ``method`` (`str`):
        Source of takeoff angles.  Only `'taup'` is currently implemented.

        ``taup_model`` (`str`):
        Name of the TauP model used for takeoff angles

        ``lambda_pol``, ``lambda_pamp``, ``lambda_sprat`` (`float`):
        Weights on the polarity, P-amplitude and S/P-ratio terms.  A weight
        of zero disables the corresponding term entirely.

        ``rho``, ``vp``, ``vs`` (`float`):
        Density (kg/m^3) and P- and S-wave speeds (m/s) of the homogeneous
        source region.  If left as `None`, each is read from `taup_model`
        at the source depth, so that the medium and the takeoff angles come
        from the same velocity model.  Any of the three can be given
        explicitly to override the model, which is what reproducing a
        published inversion generally requires.

        ``norm`` (`str`):
        `'L1'` (default) and `'L2'` are accepted.

        ``radiation_floor`` (`float`):
        Relative floor applied to `|R_P|` inside the logarithm only, so that
        a trial source with a node at a station gives a large but finite
        S/P residual instead of a numerical singularity.  Expressed as a
        fraction of the Frobenius norm of the moment tensor.

        """
        if method != 'taup':
            raise TypeError("Bad parameter: method. Only 'taup' is "
                "currently implemented")

        if norm not in ('L1', 'L2'):
            raise ValueError("Bad parameter: norm. Expected 'L1' or 'L2'")

        for name, value in (('rho', rho), ('vp', vp), ('vs', vs)):
            if value is None:
                continue
            if not np.isfinite(value) or value <= 0.:
                raise ValueError('Bad parameter: %s must be a positive '
                    'number in SI units, or None to take the value from '
                    'the velocity model at the source depth' % name)

        self.method = method
        self.taup_model = taup_model
        self.lambda_pol = float(lambda_pol)
        self.lambda_pamp = float(lambda_pamp)
        self.lambda_sprat = float(lambda_sprat)
        self.rho = None if rho is None else float(rho)
        self.vp = None if vp is None else float(vp)
        self.vs = None if vs is None else float(vs)
        self.norm = norm
        self.radiation_floor = float(radiation_floor)

        self._taup = TauPyModel(self.taup_model)


    def __call__(self, data, greens, sources, progress_handle=Null(),
            set_attributes=False):

        _check(greens, self.method)

        pol_obs, amp_obs, rat_obs, tau_pol, tau_amp, tau_rat = \
            self.get_observed(data)

        if len(pol_obs) != len(greens):
            raise ValueError('Inconsistent dimensions: %d observations but '
                '%d Green\'s tensors. Observation arrays are ordered by '
                'station, so they must correspond one-to-one with the '
                'Green\'s tensors passed to the misfit function.'
                % (len(pol_obs), len(greens)))

        pol_theo, amp_theo, log_rat_theo = self.get_predicted(greens, sources)

        n_sources = pol_theo.shape[0]
        values = np.zeros(n_sources)

        if self.lambda_pol != 0.:
            values += self.lambda_pol*_misfit_term(
                pol_obs, pol_theo, tau_pol, self.norm, 'polarity', scale=0.5)

        if self.lambda_pamp != 0.:
            values += self.lambda_pamp*_misfit_term(
                amp_obs, amp_theo, tau_amp, self.norm, 'P amplitude')

        if self.lambda_sprat != 0.:
            mask = np.isfinite(rat_obs) & (tau_rat > 0.) & (rat_obs > 0.)
            log_rat_obs = np.full(len(rat_obs), np.nan)
            log_rat_obs[mask] = np.log10(rat_obs[mask])

            values += self.lambda_sprat*_misfit_term(
                log_rat_obs, log_rat_theo, tau_rat, self.norm,
                'S/P amplitude ratio', allow_zero=True)

        # returns a NumPy array of shape (len(sources), 1)
        return values.reshape(n_sources, 1)


    def get_observed(self, data):
        """ Unpacks observed polarities, P amplitudes and S/P ratios

        Accepts either a six-element bundle
        `(pol, amp, rat, tau_pol, tau_amp, tau_rat)` as returned by
        `polarities_pamp_spamp_ratio_from_dict`, or a three-element bundle
        `(pol, amp, rat)`, in which case unit weights are assumed.
        """

        if isinstance(data, mtuq.Dataset):
            raise NotImplementedError('Measuring P amplitudes and S/P ratios '
                'from a Dataset is not implemented. These observables require '
                'windowed measurements that MTUQ does not perform; supply '
                'them with polarities_pamp_spamp_ratio_from_dict instead.')

        if not isinstance(data, (tuple, list)) or len(data) not in (3, 6):
            raise TypeError('Expected a three- or six-element bundle of '
                'observation arrays')

        arrays = [np.atleast_1d(np.asarray(item, dtype=np.float64))
            for item in data]

        lengths = set(len(item) for item in arrays)
        if len(lengths) != 1:
            raise ValueError('Inconsistent dimensions: observation arrays '
                'must all have the same length')

        if len(arrays) == 3:
            unit = np.ones(len(arrays[0]))
            arrays += [unit.copy(), unit.copy(), unit.copy()]

        return tuple(arrays)


    def _medium(self, greens):
        """ Returns the (rho, vp, vs) used in the amplitude calculations

        Values supplied to the constructor take precedence.  Any left
        unspecified are read from the velocity model at the source depth,
        so that the medium is consistent with the takeoff angles.
        """

        rho, vp, vs = self.rho, self.vp, self.vs

        if None not in (rho, vp, vs):
            return rho, vp, vs

        depth_in_m = _origin_depth(greens)
        derived = _medium_from_taup(self._taup, depth_in_m)

        return (
            derived[0] if rho is None else rho,
            derived[1] if vp is None else vp,
            derived[2] if vs is None else vs)


    def get_predicted(self, greens, sources):
        """ Calculates predicted polarities, P amplitudes and S/P ratios
        """

        if type(sources) == mtuq.MomentTensor:
            mt_array = sources.as_vector().reshape((1, 6))

        elif type(sources) == mtuq.Force:
            raise NotImplementedError

        elif _type(sources.dims) == 'MomentTensor':
            mt_array = _to_array(sources)

        elif _type(sources.dims) == 'Force':
            raise NotImplementedError

        else:
            raise TypeError

        # geometry depends only on the origin and stations, so it is
        # evaluated once here rather than once per trial source
        azimuths = _get_azimuths(greens)
        takeoff_p = _takeoff_angles_taup(
            self._taup, greens, phase_list=['p', 'P'])
        takeoff_s = _takeoff_angles_taup(
            self._taup, greens, phase_list=['s', 'S'])
        distances = _hypocentral_distances(greens)

        rho, vp, vs = self._medium(greens)

        radiation_p = _radiation_coef_P(mt_array, takeoff_p, azimuths)
        radiation_sv, radiation_sh = _radiation_coef_S(
            mt_array, takeoff_s, azimuths)

        polarities = np.sign(radiation_p)

        amplitudes = radiation_p/(
            4.*np.pi*rho*vp**3*distances[None, :])

        radiation_s = np.sqrt(radiation_sv**2 + radiation_sh**2)

        # a node in P would otherwise give an infinite ratio; the floor is
        # applied only inside the logarithm and scales with the source, so
        # it does not couple the ratio to the moment magnitude
        floor = self.radiation_floor*np.linalg.norm(
            mt_array, axis=1)[:, None]

        log_ratios = (
            np.log10(np.maximum(radiation_s, floor)) -
            np.log10(np.maximum(np.abs(radiation_p), floor)) +
            3.*np.log10(vp/vs))

        return polarities, amplitudes, log_ratios


    def collect_attributes(self, data, greens):
        """ Collects station attributes (used for beachball plots)
        """

        pol_obs, amp_obs, rat_obs = self.get_observed(data)[:3]

        azimuths = _get_azimuths(greens)
        takeoff_p = _takeoff_angles_taup(
            self._taup, greens, phase_list=['p', 'P'])
        takeoff_s = _takeoff_angles_taup(
            self._taup, greens, phase_list=['s', 'S'])

        attrs_list = []
        for _i, greens_tensor in enumerate(greens):
            attrs = AttribDict()

            try:
                attrs.azimuth = greens_tensor.azimuth
                attrs.distance_in_m = greens_tensor.distance_in_m
            except:
                pass
            try:
                attrs.network = greens_tensor.station.network
                attrs.station = greens_tensor.station.station
                attrs.location = greens_tensor.station.location
                attrs.latitude = greens_tensor.station.latitude
                attrs.longitude = greens_tensor.station.longitude
            except:
                pass
            try:
                attrs.takeoff_angle = takeoff_p[_i]
                attrs.takeoff_angle_S = takeoff_s[_i]
            except:
                pass
            try:
                attrs.polarity = pol_obs[_i]
                attrs.p_amplitude = amp_obs[_i]
                attrs.sp_ratio = rat_obs[_i]
            except:
                pass

            attrs_list += [attrs]

        return attrs_list


    def description(self):
        _description = '\n'.join([
            f'    Misfit function type:\n    {type(self).__name__}\n',
            f'    Misfit function method:\n    {self.method}\n',
            f'    Weights (polarity, P amplitude, S/P ratio):\n'
            f'    {self.lambda_pol}, {self.lambda_pamp}, '
            f'{self.lambda_sprat}\n',
            ])
        return _description



#
# misfit evaluation
#

def _misfit_term(observed, predicted, weights, norm, label, scale=1.,
        allow_zero=False):

    valid = (np.isfinite(observed) & np.isfinite(weights) & (weights > 0.))
    if not allow_zero:
        valid &= (observed != 0.)

    if not valid.any():
        raise ValueError('No usable %s observations. Either supply data for '
            'at least one station or disable this term by setting its weight '
            'to zero.' % label)

    obs = observed[valid]
    syn = predicted[:, valid]
    tau = weights[valid]

    residual = scale*(syn - obs[None, :])

    if norm == 'L1':
        denominator = np.sum(tau*np.abs(obs))
        numerator = np.sum(tau[None, :]*np.abs(residual), axis=1)
    else:
        denominator = np.sum(tau*obs**2)
        numerator = np.sum(tau[None, :]*residual**2, axis=1)

    if denominator == 0.:
        raise ValueError('Degenerate %s observations: the normalizing sum is '
            'zero, so the term cannot be evaluated. This happens when every '
            'observation is zero.' % label)

    return numerator/denominator



#
# radiation patterns
#

def _radiation_coef_S(mt_array, takeoff, azimuth):
    """ Calculates far-field SV and SH radiation coefficients

    Returns two arrays of shape `(len(mt_array), len(takeoff))`.  Angles are
    the direct-S takeoff angle measured from the downward vertical and the
    source-to-receiver azimuth, both in degrees.
    """

    n1, n2 = mt_array.shape
    if n2 != 6:
        raise Exception('Inconsistent dimensions')

    n3, n4 = len(takeoff), len(azimuth)
    if n3 != n4:
        raise Exception('Inconsistent dimensions')

    # moment tensors are stored up-south-east; the radiation geometry below
    # is written north-east-down
    m_ned = _use_to_ned(mt_array)

    takeoff = np.deg2rad(takeoff)
    azimuth = np.deg2rad(azimuth)

    rsv = np.zeros((n1, n3))
    rsh = np.zeros((n1, n3))

    for _i, (theta, phi) in enumerate(zip(takeoff, azimuth)):
        sth, cth = np.sin(theta), np.cos(theta)
        sphi, cphi = np.sin(phi), np.cos(phi)

        # Aki & Richards 2ed, p. 108
        gamma = np.array([sth*cphi, sth*sphi, cth])
        p_hat = np.array([cth*cphi, cth*sphi, -sth])
        phi_hat = np.array([-sphi, cphi, 0.])

        m_gamma = np.einsum('nab,b->na', m_ned, gamma)

        rsv[:, _i] = np.einsum('na,a->n', m_gamma, p_hat)
        rsh[:, _i] = np.einsum('na,a->n', m_gamma, phi_hat)

    return rsv, rsh


def _use_to_ned(mt_array):
    """ Converts up-south-east vectors to north-east-down matrices
    """

    mrr = mt_array[:, 0]
    mtt = mt_array[:, 1]
    mpp = mt_array[:, 2]
    mrt = mt_array[:, 3]
    mrp = mt_array[:, 4]
    mtp = mt_array[:, 5]

    m_ned = np.empty((len(mt_array), 3, 3))
    m_ned[:, 0, 0] = mtt
    m_ned[:, 1, 1] = mpp
    m_ned[:, 2, 2] = mrr
    m_ned[:, 0, 1] = m_ned[:, 1, 0] = -mtp
    m_ned[:, 0, 2] = m_ned[:, 2, 0] = mrt
    m_ned[:, 1, 2] = m_ned[:, 2, 1] = -mrp

    return m_ned


def _origin_depth(greens):
    """ Returns the source depth in m, checking that it is well defined
    """

    depths = set()
    for greens_tensor in greens:
        depths.add(float(greens_tensor.origin.depth_in_m))

    if len(depths) != 1:
        raise ValueError('Inconsistent dimensions: Green\'s tensors carry '
            'more than one origin depth')

    return depths.pop()


def _medium_from_taup(taup, depth_in_m):
    """ Reads density and wave speeds from the velocity model

    Returns `(rho, vp, vs)` in SI units, evaluated at `depth_in_m`.  The
    values are constant with respect to the receivers, so the whole-space
    approximation is preserved; only the choice of the constant is tied to
    the velocity model rather than left to the caller.
    """

    velocity_model = taup.model.s_mod.v_mod
    depth_in_km = depth_in_m/1000.

    try:
        rho = float(velocity_model.evaluate_below(depth_in_km, 'd')[0])
        vp = float(velocity_model.evaluate_below(depth_in_km, 'p')[0])
        vs = float(velocity_model.evaluate_below(depth_in_km, 's')[0])
    except Exception:
        raise ValueError('Could not read density and wave speeds from the '
            'velocity model at %.3f km depth. Supply rho, vp and vs '
            'explicitly instead.' % depth_in_km)

    if not all(np.isfinite(value) and value > 0. for value in (rho, vp, vs)):
        raise ValueError('The velocity model gives a nonpositive density or '
            'wave speed at %.3f km depth. Supply rho, vp and vs explicitly '
            'instead.' % depth_in_km)

    # velocity models are tabulated in km/s and g/cm^3
    return 1000.*rho, 1000.*vp, 1000.*vs


def _hypocentral_distances(greens):
    """ Calculates straight-line source-receiver distances

    The homogeneous whole-space approximation makes the geometrical
    spreading distance the straight line between hypocenter and receiver,
    not a ray-path length.  Receivers are assumed to lie at the surface.
    """

    distances = np.zeros(len(greens))

    for _i, greens_tensor in enumerate(greens):
        surface = greens_tensor.distance_in_m
        depth = greens_tensor.origin.depth_in_m
        distances[_i] = np.sqrt(surface**2 + depth**2)

    if np.any(distances <= 0.):
        raise ValueError('Zero source-receiver distance: the far-field '
            'approximation requires a receiver outside the source region')

    return distances



#
# input preparation
#

def polarities_pamp_spamp_ratio_from_dict(dict_polarity, dict_pamp,
    dict_spamp_ratio, stations, dict_tau_polarity=None, dict_tau_pamp=None,
    dict_tau_spamp_ratio=None, spamp_ratio_is_log10=False):

    """ Converts dictionaries of observations to arrays ordered by station

    Each dictionary is keyed by station code, optionally prefixed by network
    (`'AK.BMR'` is tried before `'BMR'`).  Stations absent from a dictionary
    are set to zero, which marks them unused for that observable.

    ``spamp_ratio_is_log10`` (`bool`):
    Declares the units of `dict_spamp_ratio`.  `False` (the default) means
    linear ratios; `True` means their base-10
    logarithms. 

    Returns a six-element bundle suitable for passing to
    `PolarityPampSPampRatioMisfit` as its `data` argument.
    """

    n = len(stations)
    pol = np.zeros(n)
    amp = np.zeros(n)
    rat = np.zeros(n)
    tau_pol = np.ones(n)
    tau_amp = np.ones(n)
    tau_rat = np.ones(n)

    for _i, station in enumerate(stations):
        pol[_i] = _lookup(dict_polarity, station, 'polarities')
        amp[_i] = _lookup(dict_pamp, station, 'P amplitudes')
        rat[_i] = _lookup(dict_spamp_ratio, station, 'S/P ratios')

        if dict_tau_polarity is not None:
            tau_pol[_i] = _lookup(
                dict_tau_polarity, station, 'polarity weights')
        if dict_tau_pamp is not None:
            tau_amp[_i] = _lookup(
                dict_tau_pamp, station, 'P amplitude weights')
        if dict_tau_spamp_ratio is not None:
            tau_rat[_i] = _lookup(
                dict_tau_spamp_ratio, station, 'S/P ratio weights')

    if spamp_ratio_is_log10:
        # convert to the linear ratios the misfit function expects; zero
        # marks a station as unused, so it is left alone
        mask = np.isfinite(rat) & (rat != 0.)
        rat[mask] = 10.**rat[mask]

    elif np.any(rat < 0.):
        raise ValueError('Negative S/P amplitude ratio. Linear ratios are '
            'positive by construction, so a negative value usually means '
            'the measurements are logarithms. Pass '
            'spamp_ratio_is_log10=True if so.')

    return pol, amp, rat, tau_pol, tau_amp, tau_rat


def _lookup(dictionary, station, label):
    """ Retrieves a station value, trying 'NET.STA' before 'STA'
    """

    if dictionary is None:
        return 0.

    for key in ('%s.%s' % (station.network, station.station), station.station):
        if key in dictionary:
            return float(dictionary[key])

    warn('Station %s not found in dictionary of %s'
        % (station.station, label))
    return 0.
