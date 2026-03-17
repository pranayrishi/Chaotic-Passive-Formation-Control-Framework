"""Gravitational and non-gravitational perturbation accelerations.

Implements J2/J3/J4 zonal harmonics, free-molecular drag, SRP with shadow,
and NRLMSISE-00 atmospheric density.
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

def get_atmospheric_density(r_eci, epoch_dt, t_since_epoch=0.0):
    """Get atmospheric mass density from NRLMSISE-00.

    Parameters
    ----------
    r_eci : ndarray, shape (3,)
        ECI position [m].
    epoch_dt : datetime
        Mission epoch (UTC).
    t_since_epoch : float
        Seconds since epoch.

    Returns
    -------
    float
        Total mass density [kg/m^3].
    """
    from nrlmsise00 import msise_model

    current_time = epoch_dt + timedelta(seconds=t_since_epoch)
    lat_deg, lon_deg, alt_km = _eci_to_geodetic(r_eci, current_time)

    # Clamp altitude to valid range
    alt_km = max(alt_km, 80.0)
    alt_km = min(alt_km, 1000.0)

    result = msise_model(current_time, alt_km, lat_deg, lon_deg,
                         F107A_NOMINAL, F107_NOMINAL, AP_NOMINAL)

    # msise_model returns a list of lists; result[0][5] is total mass density [g/cm^3]
    rho_g_cm3 = result[0][5]

    # Convert to kg/m^3
    rho = rho_g_cm3 * 1e3  # g/cm^3 -> kg/m^3

    return rho
