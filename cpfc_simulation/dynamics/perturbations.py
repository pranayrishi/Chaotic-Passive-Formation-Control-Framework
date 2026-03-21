"""Gravitational and non-gravitational perturbation accelerations.

Implements J2/J3/J4 zonal harmonics, free-molecular drag, SRP with shadow,
NRLMSISE-00 atmospheric density, USSA76 fallback atmosphere, density
uncertainty modeling (Vallado & Finkleman 2014), and beta angle / eclipse
fraction computation (Morsch Filho et al. 2020).
"""
import numpy as np
from scipy.special import erf
from datetime import timedelta
from cpfc_simulation.config import (
    MU_EARTH, R_EARTH, J2, J3, J4, OMEGA_EARTH, P_SRP, AU, CR, A_REF,
    MASS_SAT, F107_NOMINAL, F107A_NOMINAL, AP_NOMINAL, EPOCH
)


# ---------------------------------------------------------------------------
# Zonal harmonic accelerations (ECI)
# ---------------------------------------------------------------------------

def accel_J2(r):
    """J2 zonal harmonic acceleration in ECI frame.

    Parameters
    ----------
    r : array-like, shape (3,)
        ECI position [m].

    Returns
    -------
    ndarray, shape (3,)
        Acceleration [m/s^2].
    """
    x, y, z = r
    R = np.linalg.norm(r)
    factor = 1.5 * J2 * MU_EARTH * R_EARTH**2 / R**5
    z2_R2 = z**2 / R**2
    return factor * np.array([x * (5 * z2_R2 - 1),
                               y * (5 * z2_R2 - 1),
                               z * (5 * z2_R2 - 3)])


def accel_J3(r):
    """J3 zonal harmonic acceleration (Battin 1999).

    Parameters
    ----------
    r : array-like, shape (3,)
        ECI position [m].

    Returns
    -------
    ndarray, shape (3,)
        Acceleration [m/s^2].
    """
    x, y, z = r
    R = np.linalg.norm(r)
    factor = 2.5 * J3 * MU_EARTH * R_EARTH**3 / R**7
    return factor * np.array([x * (7 * z**3 / R**2 - 3 * z),
                               y * (7 * z**3 / R**2 - 3 * z),
                               7 * z**4 / R**2 - 6 * z**2 + 0.6 * R**2])


def accel_J4(r):
    """J4 zonal harmonic acceleration (Montenbruck & Gill 2000).

    Parameters
    ----------
    r : array-like, shape (3,)
        ECI position [m].

    Returns
    -------
    ndarray, shape (3,)
        Acceleration [m/s^2].
    """
    x, y, z = r
    R = np.linalg.norm(r)
    s = z / R
    factor = 0.625 * J4 * MU_EARTH * R_EARTH**4 / R**6
    common = 3 - 42 * s**2 + 63 * s**4
    return factor * np.array([x / R**2 * common,
                               y / R**2 * common,
                               z / R**2 * (15 - 70 * s**2 + 63 * s**4)])


# ---------------------------------------------------------------------------
# Free-molecular drag coefficient (Sentman 1961 / Schaaf-Chambre)
# ---------------------------------------------------------------------------

def Cd_freemolecular(alpha, s, Tw=300.0, Ti=1000.0):
    """Sentman (1961) free-molecular drag coefficient for a flat plate.

    Parameters
    ----------
    alpha : float
        Angle of attack [rad] (angle between surface normal and flow).
    s : float
        Molecular speed ratio  v_bulk / v_thermal.
    Tw : float
        Wall temperature [K].
    Ti : float
        Incident gas temperature [K].

    Returns
    -------
    float
        Drag coefficient Cd.
    """
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)

    # Schaaf-Chambre decomposition
    gamma = s * cos_a  # component of speed ratio normal to surface
    sqrt_pi = np.sqrt(np.pi)

    # P and G functions (Sentman notation)
    P = np.exp(-gamma**2) + sqrt_pi * gamma * (1.0 + erf(gamma))
    Q = 0.5 + gamma**2  # coefficient for momentum transfer

    sigma_n = 1.0  # fully diffuse accommodation

    # Pressure coefficient: normal momentum transfer
    Cp = (1.0 / (s**2 * sqrt_pi)) * (
        gamma * sqrt_pi * (1.0 + erf(gamma)) * (Q + 0.5 * sigma_n * np.sqrt(Tw / Ti))
        + np.exp(-gamma**2) * (1.0 + 0.5 * sigma_n * np.sqrt(Tw / Ti))
    )

    # Shear coefficient: tangential momentum transfer
    Ctau = (sin_a / (s * sqrt_pi)) * P * sigma_n

    # Total drag coefficient in flow direction
    Cd = Cp * cos_a + Ctau * sin_a

    # Ensure positive
    return max(Cd, 0.0)


# ---------------------------------------------------------------------------
# Atmospheric drag acceleration
# ---------------------------------------------------------------------------

def accel_drag(r_eci, v_eci, Cd, A_cross, mass, rho):
    """Atmospheric drag acceleration in ECI frame.

    Parameters
    ----------
    r_eci : ndarray, shape (3,)
        ECI position [m].
    v_eci : ndarray, shape (3,)
        ECI velocity [m/s].
    Cd : float
        Drag coefficient.
    A_cross : float
        Cross-sectional area [m^2].
    mass : float
        Satellite mass [kg].
    rho : float
        Atmospheric density [kg/m^3].

    Returns
    -------
    ndarray, shape (3,)
        Drag acceleration [m/s^2].
    """
    # Velocity relative to co-rotating atmosphere
    omega_vec = np.array([0.0, 0.0, OMEGA_EARTH])
    v_rel = v_eci - np.cross(omega_vec, r_eci)
    v_rel_mag = np.linalg.norm(v_rel)

    if v_rel_mag < 1e-10:
        return np.zeros(3)

    # Drag acceleration (opposes relative velocity)
    a_drag = -0.5 * rho * Cd * (A_cross / mass) * v_rel_mag * v_rel
    return a_drag


# ---------------------------------------------------------------------------
# Solar radiation pressure (cannonball with cylindrical shadow)
# ---------------------------------------------------------------------------

def _in_shadow(r_eci, r_sun):
    """Cylindrical Earth shadow model.

    Returns True if the satellite is in Earth's shadow.
    """
    # Vector from Sun to satellite
    r_sat_sun = r_eci - r_sun
    # Project satellite position onto Sun direction
    sun_hat = r_sun / np.linalg.norm(r_sun)
    proj = np.dot(r_eci, sun_hat)

    if proj > 0:
        # Satellite is on the sunlit side
        return False

    # Perpendicular distance from Sun-Earth line
    perp = r_eci - proj * sun_hat
    perp_dist = np.linalg.norm(perp)

    return perp_dist < R_EARTH


def accel_SRP(r_eci, r_sun, A_ref=A_REF, CR_val=CR, mass=MASS_SAT):
    """Cannonball SRP acceleration with cylindrical shadow model.

    Parameters
    ----------
    r_eci : ndarray, shape (3,)
        Satellite ECI position [m].
    r_sun : ndarray, shape (3,)
        Sun ECI position [m].
    A_ref : float
        Reference area [m^2].
    CR_val : float
        Radiation pressure coefficient (1 + reflectivity).
    mass : float
        Satellite mass [kg].

    Returns
    -------
    ndarray, shape (3,)
        SRP acceleration [m/s^2].
    """
    if _in_shadow(r_eci, r_sun):
        return np.zeros(3)

    # Vector from Sun to satellite
    d = r_eci - r_sun
    d_norm = np.linalg.norm(d)
    d_hat = d / d_norm

    # SRP scales as (AU/d)^2
    flux_scale = (AU / d_norm) ** 2

    a_srp = P_SRP * CR_val * (A_ref / mass) * flux_scale * d_hat
    return a_srp


# ---------------------------------------------------------------------------
# ECI to geodetic conversion (iterative)
# ---------------------------------------------------------------------------

def _eci_to_geodetic(r_eci, epoch_dt):
    """Convert ECI position to geodetic (lat, lon, alt).

    Uses a simple iterative algorithm (Bowring's method, 1 iteration).

    Parameters
    ----------
    r_eci : ndarray, shape (3,)
        ECI position [m].
    epoch_dt : datetime
        Current UTC datetime.

    Returns
    -------
    lat_deg : float
        Geodetic latitude [deg].
    lon_deg : float
        Geodetic longitude [deg].
    alt_km : float
        Altitude above ellipsoid [km].
    """
    # WGS84 parameters
    a_wgs = 6378137.0       # semi-major axis [m]
    f = 1.0 / 298.257223563
    e2 = 2 * f - f**2

    x, y, z = r_eci

    # Greenwich Mean Sidereal Time (approximate)
    # J2000 epoch: 2000-01-01 12:00:00 UTC
    from datetime import datetime as _dt
    j2000 = _dt(2000, 1, 1, 12, 0, 0)
    dt_sec = (epoch_dt - j2000).total_seconds()
    T = dt_sec / (36525.0 * 86400.0)  # Julian centuries

    # GMST in degrees (IAU 1982)
    gmst_deg = (280.46061837 + 360.98564736629 * (dt_sec / 86400.0)
                + 0.000387933 * T**2 - T**3 / 38710000.0) % 360.0
    gmst_rad = np.radians(gmst_deg)

    # Rotate ECI to ECEF
    cos_g = np.cos(gmst_rad)
    sin_g = np.sin(gmst_rad)
    x_ecef = cos_g * x + sin_g * y
    y_ecef = -sin_g * x + cos_g * y
    z_ecef = z

    # Geodetic coordinates (Bowring iterative)
    p = np.sqrt(x_ecef**2 + y_ecef**2)
    lon = np.arctan2(y_ecef, x_ecef)

    # Initial estimate
    lat = np.arctan2(z_ecef, p * (1.0 - e2))

    # Iterate (2 iterations is sufficient for < 1 mm accuracy)
    for _ in range(3):
        sin_lat = np.sin(lat)
        N = a_wgs / np.sqrt(1.0 - e2 * sin_lat**2)
        lat = np.arctan2(z_ecef + e2 * N * sin_lat, p)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = a_wgs / np.sqrt(1.0 - e2 * sin_lat**2)

    if abs(cos_lat) > 1e-10:
        alt = p / cos_lat - N
    else:
        alt = abs(z_ecef) - N * (1.0 - e2)

    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)
    alt_km = alt / 1000.0

    return lat_deg, lon_deg, alt_km


# ---------------------------------------------------------------------------
# NRLMSISE-00 atmospheric density
# ---------------------------------------------------------------------------

def get_atmospheric_density(r_eci, epoch_dt, t_since_epoch=0.0, F107=None, Ap=None):
    """Get atmospheric mass density from NRLMSISE-00.

    Parameters
    ----------
    r_eci : ndarray, shape (3,)
        ECI position [m].
    epoch_dt : datetime
        Mission epoch (UTC).
    t_since_epoch : float
        Seconds since epoch.
    F107 : float or None
        Daily F10.7 solar flux [SFU]. If None, uses F107_NOMINAL from config.
    Ap : float or None
        Geomagnetic Ap index. If None, uses AP_NOMINAL from config.

    Returns
    -------
    float
        Total mass density [kg/m^3].
    """
    from nrlmsise00 import msise_model

    if F107 is None:
        F107 = F107_NOMINAL
    if Ap is None:
        Ap = AP_NOMINAL

    current_time = epoch_dt + timedelta(seconds=t_since_epoch)
    lat_deg, lon_deg, alt_km = _eci_to_geodetic(r_eci, current_time)

    # Clamp altitude to valid range
    alt_km = max(alt_km, 80.0)
    alt_km = min(alt_km, 1000.0)

    result = msise_model(current_time, alt_km, lat_deg, lon_deg,
                         F107A_NOMINAL, F107, Ap)

    # msise_model returns a list of lists; result[0][5] is total mass density [g/cm^3]
    rho_g_cm3 = result[0][5]

    # Convert to kg/m^3
    rho = rho_g_cm3 * 1e3  # g/cm^3 -> kg/m^3

    return rho


# ---------------------------------------------------------------------------
# Density uncertainty model (Vallado & Finkleman 2014, Acta Astro. 95:141)
# ---------------------------------------------------------------------------

# Fractional uncertainty levels (from S-Net config / Vallado 2014)
DENSITY_UNCERT_LONGTERM  = 0.15   # 15%  long-term mean bias
DENSITY_UNCERT_SHORTTERM = 1.00   # 100% short-term worst-case
DENSITY_UNCERT_LEVELS    = [-1.00, -0.15, 0.00, +0.15, +1.00]


def apply_density_uncertainty(rho_nominal, uncertainty_fraction):
    """Apply density uncertainty to nominal NRLMSISE-00 / USSA76 value.

    Follows Vallado & Finkleman (2014): rho_actual = rho_nominal * (1 + frac).

    Parameters
    ----------
    rho_nominal : float
        Nominal density from atmospheric model [kg/m^3].
    uncertainty_fraction : float
        Fractional deviation, e.g. -0.15 (15% lower), 0.0 (nominal),
        +1.0 (100% higher).

    Returns
    -------
    float
        Adjusted density [kg/m^3].
    """
    return rho_nominal * (1.0 + uncertainty_fraction)


# ---------------------------------------------------------------------------
# USSA76 piecewise-exponential fallback atmosphere
# (Morsch Filho et al. 2020, Curtis 2014 Table A.1)
# ---------------------------------------------------------------------------

_USSA76_TABLE = [
    (  0,    1.225,        7.249),
    ( 25,    3.899e-2,     6.349),
    ( 30,    1.774e-2,     6.682),
    ( 40,    3.972e-3,     7.554),
    ( 50,    1.057e-3,     8.382),
    ( 60,    3.206e-4,     7.714),
    ( 70,    8.770e-5,     6.549),
    ( 80,    1.905e-5,     5.799),
    ( 90,    3.396e-6,     5.382),
    (100,    5.297e-7,     5.877),
    (110,    9.661e-8,     7.263),
    (120,    2.438e-8,     9.473),
    (130,    8.484e-9,    12.636),
    (140,    3.845e-9,    16.149),
    (150,    2.070e-9,    22.523),
    (180,    5.464e-10,   29.740),
    (200,    2.789e-10,   37.105),
    (250,    7.248e-11,   45.546),
    (300,    2.418e-11,   53.628),
    (350,    9.158e-12,   53.298),
    (400,    3.725e-12,   58.515),
    (450,    1.585e-12,   60.828),
    (500,    6.967e-13,   63.822),
    (600,    1.454e-13,   71.835),
    (700,    3.614e-14,   88.667),
    (800,    1.170e-14,  124.64),
    (900,    5.245e-15,  181.05),
    (1000,   3.019e-15,  268.00),
]

# Scale factor to approximate solar-minimum NRLMSISE-00 from USSA76
_NRLMSISE_SCALE = 0.55


def atmospheric_density_ussa76(alt_km, model='USSA76'):
    """Return atmospheric density [kg/m^3] from US Standard Atmosphere 1976.

    Piecewise-exponential model from Curtis (2014) Table A.1.
    Optionally scaled to approximate NRLMSISE-00 at solar minimum.

    Parameters
    ----------
    alt_km : float
        Altitude above Earth surface [km].
    model : str
        'USSA76' for raw values, 'NRLMSISE00' to apply scaling factor.

    Returns
    -------
    float
        Density [kg/m^3].
    """
    if alt_km < 0:
        return _USSA76_TABLE[0][1]
    if alt_km >= 1000:
        h0, rho0, H = _USSA76_TABLE[-1]
        rho = rho0 * np.exp(-(alt_km - h0) / H)
    else:
        rho = _USSA76_TABLE[0][1]
        for i in range(len(_USSA76_TABLE) - 1):
            h0, rho0, H = _USSA76_TABLE[i]
            h1 = _USSA76_TABLE[i + 1][0]
            if h0 <= alt_km < h1:
                rho = rho0 * np.exp(-(alt_km - h0) / H)
                break

    if model == 'NRLMSISE00':
        rho *= _NRLMSISE_SCALE
    return rho


def get_atmospheric_density_with_fallback(r_eci, epoch_dt, t_since_epoch=0.0,
                                          F107=None, Ap=None):
    """Get atmospheric density, falling back to USSA76 if NRLMSISE-00 unavailable.

    Parameters
    ----------
    r_eci : ndarray, shape (3,)
        ECI position [m].
    epoch_dt : datetime
        Mission epoch (UTC).
    t_since_epoch : float
        Seconds since epoch.
    F107, Ap : float or None
        Solar/geomagnetic indices.

    Returns
    -------
    rho : float
        Atmospheric density [kg/m^3].
    source : str
        'NRLMSISE00' or 'USSA76'.
    """
    try:
        rho = get_atmospheric_density(r_eci, epoch_dt, t_since_epoch, F107, Ap)
        return rho, 'NRLMSISE00'
    except (ImportError, Exception):
        # Fall back to USSA76
        current_time = epoch_dt + timedelta(seconds=t_since_epoch)
        _, _, alt_km = _eci_to_geodetic(r_eci, current_time)
        alt_km = max(alt_km, 0.0)
        rho = atmospheric_density_ussa76(alt_km)
        return rho, 'USSA76'


# ---------------------------------------------------------------------------
# Beta angle and eclipse fraction (Morsch Filho et al. 2020, Eqs. 39-41)
# ---------------------------------------------------------------------------

def beta_angle(inc_rad, raan_rad, ecliptic_longitude_rad,
               obliquity_rad=np.radians(23.4393)):
    """Compute orbit beta angle (angle between orbit plane and Sun vector).

    Equation 39 of Morsch Filho et al. (2020).

    Parameters
    ----------
    inc_rad : float
        Orbital inclination [rad].
    raan_rad : float
        Right ascension of ascending node [rad].
    ecliptic_longitude_rad : float
        Sun ecliptic longitude [rad].
    obliquity_rad : float
        Earth's obliquity [rad].

    Returns
    -------
    float
        Beta angle [deg].
    """
    lam = ecliptic_longitude_rad
    eps = obliquity_rad
    i = inc_rad
    Omega = raan_rad

    beta = np.arcsin(
        np.cos(lam) * np.sin(Omega) * np.sin(i)
        - np.sin(lam) * np.cos(eps) * np.cos(Omega) * np.sin(i)
        + np.sin(lam) * np.sin(eps) * np.cos(i)
    )
    return np.degrees(beta)


def eclipse_fraction(beta_deg, alt_km):
    """Compute analytical eclipse fraction of an orbit.

    Equations 40-41 of Morsch Filho et al. (2020).

    Parameters
    ----------
    beta_deg : float
        Beta angle [deg].
    alt_km : float
        Orbital altitude [km].

    Returns
    -------
    float
        Eclipse fraction [0, 1].
    """
    r = R_EARTH / 1000.0 + alt_km  # [km]
    RE_km = R_EARTH / 1000.0

    # Eq. 41: critical beta angle
    beta_star_deg = np.degrees(np.arcsin(RE_km / r))

    if abs(beta_deg) >= beta_star_deg:
        return 0.0  # no eclipse

    # Eq. 40
    beta_r = np.radians(beta_deg)
    numer = np.sqrt((r - RE_km)**2 + 2 * RE_km * (r - RE_km))
    denom = r * np.cos(beta_r)
    if denom <= 0:
        return 0.0
    fE = (1.0 / 180.0) * np.degrees(np.arccos(np.clip(numer / denom, -1, 1)))
    return np.clip(fE, 0.0, 1.0)
