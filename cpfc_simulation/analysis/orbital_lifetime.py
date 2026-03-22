"""Orbital lifetime estimation for LEO CubeSats under atmospheric drag.

Implements the SMAD (Wertz & Larson) analytical decay model and a numerical
integrator, motivated by DuBois, Jacobs & Vaidyanathan (2026) analysis of
the DORA 3U CubeSat re-entry under Solar Cycle 25 conditions.

Key insight from the DORA paper: solar activity can reduce CubeSat lifetime
by up to 8x.  This module connects that finding to the CPFC framework by
computing estimated orbital lifetime at each (altitude, F10.7) point, enabling
a tradeoff analysis: higher solar activity gives more differential drag
authority for formation control BUT shortens the mission window.
"""
import numpy as np
from cpfc_simulation.config import (
    MU_EARTH, R_EARTH, MASS_SAT, A_STOWED, A_DEPLOYED,
    F107_NOMINAL, AP_NOMINAL, EPOCH,
)


# ---------------------------------------------------------------------------
#  SMAD analytical lifetime (Wertz & Larson, Ch. 8)
# ---------------------------------------------------------------------------

def smad_lifetime_estimate(h_km, Cd=2.2, A_cross=None, mass=None,
                           F107=None, Ap=None):
    """Estimate orbital lifetime using the SMAD analytical model.

    Uses an exponential atmosphere with scale height interpolated from
    the CIRA-72 / US Standard Atmosphere table, and the King-Hele secular
    decay rate.

    Parameters
    ----------
    h_km : float or ndarray
        Altitude [km].
    Cd : float
        Drag coefficient (default 2.2, same as DuBois et al.).
    A_cross : float or None
        Cross-sectional area [m^2].  Defaults to A_STOWED.
    mass : float or None
        Satellite mass [kg].  Defaults to MASS_SAT.
    F107 : float or None
        Solar 10.7 cm radio flux [SFU].  Defaults to F107_NOMINAL.
    Ap : float or None
        Geomagnetic Ap index.  Defaults to AP_NOMINAL.

    Returns
    -------
    lifetime_days : float or ndarray
        Estimated remaining orbital lifetime [days].
    """
    if A_cross is None:
        A_cross = A_STOWED
    if mass is None:
        mass = MASS_SAT
    if F107 is None:
        F107 = F107_NOMINAL
    if Ap is None:
        Ap = AP_NOMINAL

    h_km = np.asarray(h_km, dtype=float)

    # Ballistic coefficient  B = m / (Cd * A)  [kg/m^2]
    B = mass / (Cd * A_cross)

    # Atmospheric density from simple exponential model adjusted for F10.7.
    # Base densities from US Standard Atmosphere (SMAD Table 8-4), then
    # scaled by  (F10.7 / 150)^0.7  to approximate NRLMSISE-00 solar
    # dependence (Picone et al. 2002).
    rho_base = _exponential_density(h_km)
    solar_scale = (F107 / 150.0) ** 0.7
    rho = rho_base * solar_scale

    # Orbital velocity  v = sqrt(mu / r)
    r = R_EARTH + h_km * 1e3
    v = np.sqrt(MU_EARTH / r)  # [m/s]

    # King-Hele secular decay rate  dh/dt = -rho * v * (Cd * A / m) / 2
    # but expressed for lifetime:  tau ~ H / (rho * v * Cd * A / m)
    # where H = scale height.
    H = _scale_height(h_km)   # [m]

    # Lifetime = (H * B) / (rho * v * r)  (derived from dh/dt integration
    # assuming constant scale height over one scale-height drop)
    lifetime_s = H * B / (rho * v)
    lifetime_days = lifetime_s / 86400.0

    return lifetime_days


def numerical_lifetime_estimate(h_km_start, Cd=2.2, A_cross=None, mass=None,
                                F107=None, Ap=None, h_reentry_km=120.0,
                                dt_days=0.25):
    """Numerically integrate orbital decay until re-entry altitude.

    Uses Euler integration of King-Hele decay with NRLMSISE-00 density
    when available, falling back to the scaled exponential model.

    Parameters
    ----------
    h_km_start : float
        Initial altitude [km].
    Cd : float
        Drag coefficient.
    A_cross : float or None
        Cross-sectional area [m^2].
    mass : float or None
        Satellite mass [kg].
    F107 : float or None
        Solar flux [SFU].
    Ap : float or None
        Geomagnetic index.
    h_reentry_km : float
        Re-entry altitude threshold [km].
    dt_days : float
        Integration timestep [days].

    Returns
    -------
    dict
        'lifetime_days': total lifetime,
        'h_history': altitude vs time,
        't_history': time array [days].
    """
    if A_cross is None:
        A_cross = A_STOWED
    if mass is None:
        mass = MASS_SAT
    if F107 is None:
        F107 = F107_NOMINAL
    if Ap is None:
        Ap = AP_NOMINAL

    B = mass / (Cd * A_cross)
    dt_s = dt_days * 86400.0

    h_list = [h_km_start]
    t_list = [0.0]
    h = h_km_start
    t = 0.0

    # Try to use NRLMSISE-00
    use_nrlmsise = False
    try:
        from cpfc_simulation.dynamics.perturbations import get_atmospheric_density
        use_nrlmsise = True
    except ImportError:
        pass

    max_days = 10000  # safety limit
    while h > h_reentry_km and t < max_days * 86400.0:
        r = R_EARTH + h * 1e3
        v = np.sqrt(MU_EARTH / r)

        if use_nrlmsise:
            try:
                r_eci = np.array([r, 0.0, 0.0])
                rho = get_atmospheric_density(r_eci, EPOCH, t, F107=F107, Ap=Ap)
            except Exception:
                rho = _exponential_density(h) * (F107 / 150.0) ** 0.7
        else:
            rho = _exponential_density(h) * (F107 / 150.0) ** 0.7

        # dh/dt = -0.5 * rho * v * (Cd * A / m) * r  (King-Hele)
        # but in km/s units:
        dh_dt = -0.5 * rho * v * (Cd * A_cross / mass) * r  # [m/s]
        dh_km = (dh_dt * dt_s) / 1e3  # convert to km

        h += dh_km
        t += dt_s

        h_list.append(h)
        t_list.append(t)

    return {
        'lifetime_days': t / 86400.0,
        'h_history': np.array(h_list),
        't_history': np.array(t_list) / 86400.0,
    }


def lifetime_vs_solar_flux(h_km, F107_range=None, Cd=2.2, A_cross=None,
                           mass=None):
    """Compute lifetime across a range of solar flux values.

    Parameters
    ----------
    h_km : float
        Altitude [km].
    F107_range : ndarray or None
        Array of F10.7 values [SFU].
    Cd : float
        Drag coefficient.
    A_cross : float or None
        Cross-sectional area [m^2].
    mass : float or None
        Satellite mass [kg].

    Returns
    -------
    F107_range : ndarray
        Solar flux values.
    lifetimes : ndarray
        Lifetime in days for each F10.7 value.
    """
    if F107_range is None:
        F107_range = np.arange(70, 260, 10)

    lifetimes = np.array([
        smad_lifetime_estimate(h_km, Cd=Cd, A_cross=A_cross, mass=mass, F107=f)
        for f in F107_range
    ])
    return F107_range, lifetimes


def mission_feasibility(h_km, F107, mission_duration_days=30.0,
                        Cd=2.2, A_cross_stowed=None, A_cross_deployed=None,
                        mass=None):
    """Assess whether a formation mission is feasible before orbital decay.

    Computes:
    - Estimated lifetime (SMAD)
    - Time margin (lifetime - mission duration)
    - Differential drag authority (dfy_max)
    - Along-track correction per orbit

    Parameters
    ----------
    h_km : float
        Altitude [km].
    F107 : float
        Solar flux [SFU].
    mission_duration_days : float
        Planned mission duration [days].
    Cd : float
        Drag coefficient.
    A_cross_stowed : float or None
        Stowed cross-section [m^2].
    A_cross_deployed : float or None
        Deployed cross-section [m^2].
    mass : float or None
        Satellite mass [kg].

    Returns
    -------
    dict
        Feasibility assessment with lifetime, margin, and authority metrics.
    """
    if A_cross_stowed is None:
        A_cross_stowed = A_STOWED
    if A_cross_deployed is None:
        A_cross_deployed = A_DEPLOYED
    if mass is None:
        mass = MASS_SAT

    # Lifetime estimate (use stowed area — worst case for lifetime)
    lifetime_days = float(smad_lifetime_estimate(
        h_km, Cd=Cd, A_cross=A_cross_stowed, mass=mass, F107=F107
    ))

    margin_days = lifetime_days - mission_duration_days
    feasible = margin_days > 0

    # Differential drag authority
    rho_base = _exponential_density(h_km)
    solar_scale = (F107 / 150.0) ** 0.7
    rho = rho_base * solar_scale

    r = R_EARTH + h_km * 1e3
    v = np.sqrt(MU_EARTH / r)
    n = np.sqrt(MU_EARTH / r**3)
    T_orb = 2 * np.pi / n

    dfy_max = 0.5 * rho * v**2 * Cd * (A_cross_deployed - A_cross_stowed) / mass

    return {
        'h_km': h_km,
        'F107': F107,
        'lifetime_days': lifetime_days,
        'mission_duration_days': mission_duration_days,
        'margin_days': margin_days,
        'feasible': feasible,
        'dfy_max': dfy_max,
        'rho': rho,
        'T_orb_min': T_orb / 60.0,
    }


def feasibility_grid(h_range_km, F107_range, mission_duration_days=30.0,
                     Cd=2.2, mass=None):
    """Compute feasibility across altitude x solar flux grid.

    Parameters
    ----------
    h_range_km : ndarray
        Altitude range [km].
    F107_range : ndarray
        Solar flux range [SFU].
    mission_duration_days : float
        Mission duration [days].
    Cd : float
        Drag coefficient.
    mass : float or None
        Satellite mass [kg].

    Returns
    -------
    dict
        'lifetime_grid': (nh, nf) lifetime in days,
        'margin_grid': (nh, nf) margin in days,
        'feasible_grid': (nh, nf) boolean feasibility,
        'dfy_max_grid': (nh, nf) max diff drag [m/s^2].
    """
    nh = len(h_range_km)
    nf = len(F107_range)

    lifetime_grid = np.zeros((nh, nf))
    margin_grid = np.zeros((nh, nf))
    feasible_grid = np.zeros((nh, nf), dtype=bool)
    dfy_max_grid = np.zeros((nh, nf))

    for i, h in enumerate(h_range_km):
        for j, f107 in enumerate(F107_range):
            result = mission_feasibility(
                h, f107, mission_duration_days=mission_duration_days,
                Cd=Cd, mass=mass,
            )
            lifetime_grid[i, j] = result['lifetime_days']
            margin_grid[i, j] = result['margin_days']
            feasible_grid[i, j] = result['feasible']
            dfy_max_grid[i, j] = result['dfy_max']

    return {
        'h_range_km': h_range_km,
        'F107_range': F107_range,
        'lifetime_grid': lifetime_grid,
        'margin_grid': margin_grid,
        'feasible_grid': feasible_grid,
        'dfy_max_grid': dfy_max_grid,
    }


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

# Scale heights from US Standard Atmosphere / CIRA-72 (SMAD Table 8-4)
_SCALE_HEIGHT_TABLE = np.array([
    # (altitude_km, scale_height_m)
    [100,   5900],
    [150,  22000],
    [200,  29400],
    [250,  37500],
    [300,  45500],
    [350,  53500],
    [400,  58500],
    [450,  62000],
    [500,  65500],
    [550,  69000],
    [600,  72000],
    [700,  78000],
    [800,  84000],
    [900,  89000],
    [1000, 95000],
])

# Reference densities at these altitudes (US Standard Atmosphere, quiet sun)
_DENSITY_TABLE = np.array([
    # (altitude_km, density_kg_m3)
    [100, 5.297e-7],
    [150, 2.076e-9],
    [200, 2.541e-10],
    [250, 6.073e-11],
    [300, 1.916e-11],
    [350, 7.014e-12],
    [400, 2.803e-12],
    [450, 1.184e-12],
    [500, 5.215e-13],
    [550, 2.384e-13],
    [600, 1.137e-13],
    [700, 3.070e-14],
    [800, 1.136e-14],
    [900, 5.759e-15],
    [1000, 3.561e-15],
])


def _scale_height(h_km):
    """Interpolate scale height [m] from SMAD table."""
    h_km = np.asarray(h_km, dtype=float)
    return np.interp(h_km, _SCALE_HEIGHT_TABLE[:, 0], _SCALE_HEIGHT_TABLE[:, 1])


def _exponential_density(h_km):
    """Interpolate base atmospheric density [kg/m^3] from SMAD table."""
    h_km = np.asarray(h_km, dtype=float)
    alts = _DENSITY_TABLE[:, 0]
    log_rho = np.log(_DENSITY_TABLE[:, 1])
    return np.exp(np.interp(h_km, alts, log_rho))
