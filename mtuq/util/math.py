
import numpy as np
from obspy.geodetics import gps2dist_azimuth
from scipy.signal import fftconvolve


#
# numerical
#

def isclose(X, Y):
    EPSVAL = 1.e-6
    X = np.array(X)
    Y = np.array(Y)
    return bool(
        np.linalg.norm(X-Y) < EPSVAL)


def correlate(v1, v2):
    """ Fast cross-correlation function

    Correlates unpadded array v1 and padded array v2, producing result of
    shape ``len(v2) - len(v1)``
    """
    n1, n2 = len(v1), len(v2)

    if n1>2000 or n2-n1>200:
        # for long traces, frequency-domain implementation is usually faster
        return fftconvolve(v1, v2[::-1], 'valid')
    else:
        # for short traces, time-domain implementation is usually faster
        return np.correlate(v1, v2, 'valid')


def wrap_180(angle_in_deg):
    """ Wraps angle to (-180, 180)
    """
    angle_in_deg %= 360.
    idx = np.where(angle_in_deg > 180.)
    angle_in_deg[idx] -= 360.
    return angle_in_deg


#
# statistical
#

def apply_cov(C, r):
    """ Applies covariance matrix to residuals vector
    """
    #
    # TODO - more robust for different array types, more efficient
    #

    if type(C) not in [np.array, float]:
        raise Exception

    ndim = getattr(C, 'ndim', 0)

    if ndim not in [0,1,2]:
        raise Exception

    if ndim==2:
        Cinv = np.linalg.inv(C)
        return np.dot(r, np.dot(Cinv, r))

    else:
        return np.sum(r**2/C)


#
# set theoretic
#


def list_intersect(a, b):
    """ Intersection of two lists
    """
    return list(set(a).intersection(set(b)))


def list_intersect_with_indices(a, b):
    intersection = list(set(a).intersection(set(b)))
    indices = [a.index(item) for item in intersection]
    return intersection, indices


def open_interval(x1, x2, N):
    """ Covers the open interval (x1, x2) with N regularly-spaced points
    """

    # NOTE: np.linspace(x1, x2, N)[1:-1] would be slightly simpler
    # but not as readily used by matplotlib.pyplot.pcolor

    return np.linspace(x1, x2, 2*N+1)[1:-1:2]


def closed_interval(x1, x2, N):
    """ Covers the closed interval [x1, x2] with N regularly-spaced points
    """
    return np.linspace(x1, x2, N)


def tight_interval(x1,x2,N,tightness=0.999):
    # tightness (float)
    # 0. reduces to ``open_intervel``, 1. reduces to ``closed_intervel``
    Lo = open_interval(x1,x2,N)
    Lc = closed_interval(x1,x2,N)
    return Lo*(1.-tightness) + Lc*tightness



#
# moment tensor and force
#

def lune_det(delta, gamma):
    """ Determinant of lune mapping as function of lune coordinates
    """
    delta, gamma = np.meshgrid(np.deg2rad(delta), np.deg2rad(gamma))
    beta = np.pi/2. - delta
    return 4./np.pi * np.sin(beta)**3 * np.cos(3.*gamma)


def to_mij(rho, v, w, kappa, sigma, h):
    """ Converts from lune parameters to moment tensor parameters
    (up-south-east convention)
    """
    kR3 = np.sqrt(3.)
    k2R6 = 2.*np.sqrt(6.)
    k2R3 = 2.*np.sqrt(3.)
    k4R6 = 4.*np.sqrt(6.)
    k8R6 = 8.*np.sqrt(6.)

    m0 = rho/np.sqrt(2.)

    delta, gamma = to_delta_gamma(v, w)
    beta = 90. - delta

    gamma = np.deg2rad(gamma)
    beta = np.deg2rad(90. - delta)
    kappa = np.deg2rad(kappa)
    sigma = np.deg2rad(sigma)
    theta = np.arccos(h)

    Cb  = np.cos(beta)
    Cg  = np.cos(gamma)
    Cs  = np.cos(sigma)
    Ct  = np.cos(theta)
    Ck  = np.cos(kappa)
    C2k = np.cos(2.0*kappa)
    C2s = np.cos(2.0*sigma)
    C2t = np.cos(2.0*theta)

    Sb  = np.sin(beta)
    Sg  = np.sin(gamma)
    Ss  = np.sin(sigma)
    St  = np.sin(theta)
    Sk  = np.sin(kappa)
    S2k = np.sin(2.0*kappa)
    S2s = np.sin(2.0*sigma)
    S2t = np.sin(2.0*theta)

    mt0 = m0 * (1./12.) * \
        (k4R6*Cb + Sb*(kR3*Sg*(-1. - 3.*C2t + 6.*C2s*St*St) + 12.*Cg*S2t*Ss))

    mt1 = m0* (1./24.) * \
        (k8R6*Cb + Sb*(-24.*Cg*(Cs*St*S2k + S2t*Sk*Sk*Ss) + kR3*Sg * \
        ((1. + 3.*C2k)*(1. - 3.*C2s) + 12.*C2t*Cs*Cs*Sk*Sk - 12.*Ct*S2k*S2s)))

    mt2 = m0* (1./6.) * \
        (k2R6*Cb + Sb*(kR3*Ct*Ct*Ck*Ck*(1. + 3.*C2s)*Sg - k2R3*Ck*Ck*Sg*St*St +
        kR3*(1. - 3.*C2s)*Sg*Sk*Sk + 6.*Cg*Cs*St*S2k +
        3.*Ct*(-4.*Cg*Ck*Ck*St*Ss + kR3*Sg*S2k*S2s)))

    mt3 = m0* (-1./2.)*Sb*(k2R3*Cs*Sg*St*(Ct*Cs*Sk - Ck*Ss) +
        2.*Cg*(Ct*Ck*Cs + C2t*Sk*Ss))

    mt4 = -m0* (1./2.)*Sb*(Ck*(kR3*Cs*Cs*Sg*S2t + 2.*Cg*C2t*Ss) +
        Sk*(-2.*Cg*Ct*Cs + kR3*Sg*St*S2s))

    mt5 = -m0* (1./8.)*Sb*(4.*Cg*(2.*C2k*Cs*St + S2t*S2k*Ss) +
        kR3*Sg*((1. - 2.*C2t*Cs*Cs - 3.*C2s)*S2k + 4.*Ct*C2k*S2s))

    if type(mt0) is np.ndarray:
        return np.column_stack([mt0, mt1, mt2, mt3, mt4, mt5])
    else:
        return np.array([mt0, mt1, mt2, mt3, mt4, mt5])


def from_mij(mij):
    """
    Converts from moment tensor (Up-South-East) to lune parameters
    This is a stripped out version based of mtpar's cmt2tt function by rmodrak
    It ONLY works with MTUQ convention.

    Parameters:
    -----------
    mij : array_like, shape (6,)
        Moment tensor components in Up-South-East convention (default MTUQ) 
        [Mxx, Myy, Mzz, Mxy, Mxz, Myz]
        
    Returns:
    --------
    tuple : (rho, v, w, kappa, sigma, h)
        rho   : Tape2012 magnitude parameter  
        v     : Tape2015 parameter v [-1/3, 1/3]
        w     : Tape2015 parameter w [-3π/8, 3π/8]
        kappa : strike angle [0°, 360°]
        sigma : rake angle [-90°, 90°]
        h     : cosine of dip angle [0, 1]

    """
    
    mij = np.array(mij)
    
    # Cast from up-south-east to south-east-up convention (following mtpar)
    # USE convention: [Mxx, Myy, Mzz, Mxy, Mxz, Myz] (UP-SOUTH-EAST)
    # SEU convention: [Myy, Mzz, Mxx, Myz, Mxy, Mxz] (SOUTH-EAST-UP)
    mij_seu = np.array([mij[1], mij[2], mij[0], mij[5], mij[3], mij[4]])
    
    # Convert to matrix representation for eigenvalue decomposition
    M_seu = np.array([[mij_seu[0], mij_seu[3], mij_seu[4]],
                      [mij_seu[3], mij_seu[1], mij_seu[5]],
                      [mij_seu[4], mij_seu[5], mij_seu[2]]])
    
    # Eigenvalue decomposition (sort eigenvalues highest to lowest)
    lam, U = np.linalg.eigh(M_seu)
    idx = np.argsort(lam)[::-1]  # descending sort
    lam = lam[idx]
    U = U[:, idx]
    
    # Convert eigenvalues to lune coordinates
    # magnitude of lambda vector
    lammag = np.linalg.norm(lam)
    
    # seismic moment M0 = ||lam|| / sqrt(2)
    M0 = lammag / np.sqrt(2.)
    rho = M0 * np.sqrt(2.)  # rho parameter
    
    # lune coordinates (gamma, delta)
    if np.sum(lam) != 0.:
        bdot = np.sum(lam) / (np.sqrt(3) * lammag)
        bdot = np.clip(bdot, -1, 1)  # clipping to avoid numerical issues
        delta = 90. - np.rad2deg(np.arccos(bdot))
    else:
        delta = 0.
    
    # gamma coordinate
    if lam[0] != lam[2]:
        gamma = np.rad2deg(np.arctan((-lam[0] + 2.*lam[1] - lam[2]) / 
                                   (np.sqrt(3) * (lam[0] - lam[2]))))
    else:
        gamma = 0.
    
    # Convert lune coordinates to v, w parameters
    gamma_rad = np.deg2rad(gamma)
    delta_rad = np.deg2rad(delta)
    beta = np.pi/2. - delta_rad
    
    v = (1./3.) * np.sin(3. * gamma_rad)
    u = (0.75 * beta - 0.5 * np.sin(2. * beta) + 0.0625 * np.sin(4. * beta))
    w = 3. * np.pi / 8. - u
    
    # Ensure det(U) = 1
    if np.linalg.det(U) < 0:
        U[:, 1] *= -1
    
    # 45° rotation around y axis to get the fault vectors from eigenvectors
    Y = np.array([[np.cos(np.pi/4), 0, np.sin(np.pi/4)],
                  [0, 1, 0],
                  [-np.sin(np.pi/4), 0, np.cos(np.pi/4)]])  # rotmat(45, 1)
    
    V = np.dot(U, Y)
    S = V[:, 0]  # slip vector
    N = V[:, 2]  # fault normal
    
    # Round off small values to -/+1 and 0 (like in mtpar)
    EPSVAL = 1e-6
    S[np.abs(S) < EPSVAL] = 0
    N[np.abs(N) < EPSVAL] = 0
    S[np.abs(S - 1) < EPSVAL] = 1
    S[np.abs(S + 1) < EPSVAL] = -1
    N[np.abs(N - 1) < EPSVAL] = 1
    N[np.abs(N + 1) < EPSVAL] = -1
    
    # Calculate fault angles using south-east-up basis
    zenith = np.array([0, 0, 1])
    north = np.array([-1, 0, 0])
    
    def faultvec2angles(S_vec, N_vec):
        """Calculate fault angles from slip and normal vectors
        Similar to Ryan's mtpar implementation. I keep within the scope of 
        the from_mij function because it is only useful there."""
        # Strike vector
        v_cross = np.cross(zenith, N_vec)
        if np.linalg.norm(v_cross) == 0:
            K = S_vec  # horizontal fault case
        else:
            K = v_cross / np.linalg.norm(v_cross)
        
        # Strike angle (kappa)
        def fangle_signed(va, vb, vnor):
            # Angle between two vectors with sign
            xy = np.dot(va, vb)
            xx = np.dot(va, va)
            yy = np.dot(vb, vb)
            theta = np.rad2deg(np.arccos(np.clip(xy / (xx * yy)**0.5, -1, 1)))
            
            if abs(theta - 180) <= EPSVAL:
                return 180
            else:
                Dmat = np.column_stack([va, vb, vnor])
                if np.linalg.det(Dmat) < 0:
                    return -theta
                else:
                    return theta
        
        kappa = fangle_signed(north, K, -zenith)
        kappa = kappa % 360.  # wrap to [0, 360)
        
        # Dip angle (theta)
        costh = np.dot(N_vec, zenith)
        theta = np.rad2deg(np.arccos(np.clip(costh, -1, 1)))
        
        # Rake angle (sigma)
        sigma = fangle_signed(K, S_vec, N_vec)
        
        return theta, sigma, kappa, K
    
    # Frame2angles: evaluate four combinations to resolve ambiguity
    # There are four combinations of N and S that represent a double couple
    # moment tensor (TT2012, Fig. 15). We need to find the one within the
    # proper bounding region (TT2012, Figs. 16, B1)
    
    # Four combinations for the given frame
    S1, N1 = S, N
    S2, N2 = -S, -N  
    S3, N3 = N, S
    S4, N4 = -N, -S
    
    # Calculate fault angles for each combination
    theta1, sigma1, kappa1, K1 = faultvec2angles(S1, N1)
    theta2, sigma2, kappa2, K2 = faultvec2angles(S2, N2)
    theta3, sigma3, kappa3, K3 = faultvec2angles(S3, N3)
    theta4, sigma4, kappa4, K4 = faultvec2angles(S4, N4)
    
    theta = np.array([theta1, theta2, theta3, theta4])
    sigma = np.array([sigma1, sigma2, sigma3, sigma4])
    kappa = np.array([kappa1, kappa2, kappa3, kappa4])
    
    # Which combination lies within the bounding region?
    btheta = (theta <= 90. + EPSVAL)
    bsigma = (np.abs(sigma) <= 90. + EPSVAL)
    bb = np.logical_and(btheta, bsigma)
    ii = np.where(bb)[0]
    nn = len(ii)
    
    if nn == 0:
        raise Exception('No valid fault plane found within bounding region')
    elif nn == 1:
        jj = ii[0]
    elif nn == 2:
        # Choose one of the two based on strike angle
        # This is a simplified version of the _pick function in mtpar
        if kappa[ii[0]] < 180:
            jj = ii[0]
        else:
            jj = ii[1]
    else:
        # Take the first one for unusual cases
        jj = ii[0]
    
    # Select the angles from the chosen combination
    final_theta = theta[jj]
    final_sigma = sigma[jj]
    final_kappa = kappa[jj]
    
    # Convert theta to h parameter
    h = np.cos(np.deg2rad(final_theta))
    
    return (rho, v, w, final_kappa, final_sigma, h)


def to_xyz(F0, phi, h):
    """ Converts from spherical to Cartesian coordinates (east-north-up)
    """
    # spherical coordinates in "physics convention"
    r = F0
    phi = np.radians(phi)
    theta = np.arccos(h)

    x = F0*np.sin(theta)*np.cos(phi)
    y = F0*np.sin(theta)*np.sin(phi)
    z = F0*np.cos(theta)

    if type(F0) is np.ndarray:
        return np.column_stack([x, y, z])
    else:
        return np.array([x, y, z])


def to_rtp(F0, phi, h):
    """ Converts from spherical to Cartesian coordinates (up-south-east)

    Parameters:
    F0 (float or numpy array): The radial distance from the origin
    phi (float or numpy array): The azimuthal angle in degrees - 0 to 360 range, where 0 is East. Anticlockwise positive.
    h (float or numpy array): The cosine of the polar angle - -1 to 1 range, where 1 is Up
    
    Returns:
    numpy array: The Cartesian coordinates in the up-south-east system
    """
    # spherical coordinates in "physics convention"
    
    r = F0
    phi = np.radians(phi)
    theta = np.arccos(h)

    x = F0*np.sin(theta)*np.cos(phi)
    y = F0*np.sin(theta)*np.sin(phi)
    z = F0*np.cos(theta)

    if type(F0) is np.ndarray:
        return np.column_stack([z, -y, x,])
    else:
        return np.array([z, -y, x])


def to_delta_gamma(v, w):
    """ Converts from Tape2015 parameters to lune coordinates
    """
    return to_delta(w), to_gamma(v)


def to_gamma(v):
    """ Converts from Tape2015 parameter v to lune longitude
    """
    gamma = (1./3.)*np.arcsin(3.*v)
    return np.rad2deg(gamma)


def to_delta(w):
    """ Converts from Tape2015 parameter w to lune latitude
    """
    beta0 = np.linspace(0, np.pi, 100)
    u0 = 0.75*beta0 - 0.5*np.sin(2.*beta0) + 0.0625*np.sin(4.*beta0)
    beta = np.interp(3.*np.pi/8. - w, u0, beta0)
    delta = np.rad2deg(np.pi/2. - beta)
    return delta


def to_v_w(delta, gamma):
    """ Converts from lune coordinates to Tape2015 parameters
    """
    return to_v(gamma), to_w(delta)


def to_v(gamma):
    """ Converts from lune longitude to Tape2015 parameter v
    """
    v = (1./3.)*np.sin(3.*np.deg2rad(gamma))
    return v


def to_w(delta):
    """ Converts from lune latitude to Tape2015 parameter w
    """
    beta = np.deg2rad(90. - delta)
    u = (0.75*beta - 0.5*np.sin(2.*beta) + 0.0625*np.sin(4.*beta))
    w = 3.*np.pi/8. - u
    return w


def to_M0(Mw):
    """ Converts from moment magnitude to scalar moment
    """
    return 10.**(1.5*float(Mw) + 9.1)


def to_rho(Mw):
    """ Converts from moment magnitude to Tape2012 magnitude parameter
    """
    return to_M0(Mw)*np.sqrt(2.)

def to_Mw(rho):
    """ Converts from Tape2012 magnitude parameter to moment magnitude
    """
    return ((np.log10(rho/np.sqrt(2))-9.1)/1.5)


#
# structured grids
#

def lat_lon_tuples(center_lat=None,center_lon=None,
    spacing_in_m=None, npts_per_edge=None, perturb_in_deg=0.1):
    """ Geographic grid
    """

    # calculate spacing in degrees latitude
    perturb_in_m, _, _ = gps2dist_azimuth(
        center_lat,  center_lon, center_lat + perturb_in_deg, center_lon)

    spacing_lat = spacing_in_m * (perturb_in_deg/perturb_in_m)


    # calculate spacing in degrees longitude
    perturb_in_m, _, _ = gps2dist_azimuth(
        center_lat,  center_lon, center_lat, center_lon + perturb_in_deg)

    spacing_lon = spacing_in_m * (perturb_in_deg/perturb_in_m)

    edge_length_lat = spacing_lat*(npts_per_edge-1)
    edge_length_lon = spacing_lon*(npts_per_edge-1)


    # construct regularly-spaced grid
    lat_vec = np.linspace(
        center_lat-edge_length_lat/2., center_lat+edge_length_lat/2., npts_per_edge)

    lon_vec = np.linspace(
        center_lon-edge_length_lon/2., center_lon+edge_length_lon/2., npts_per_edge)

    lat, lon = np.meshgrid(lat_vec, lon_vec)

    # return tuples
    lat = lat.flatten()
    lon = lon.flatten()

    return zip(lat, lon)


