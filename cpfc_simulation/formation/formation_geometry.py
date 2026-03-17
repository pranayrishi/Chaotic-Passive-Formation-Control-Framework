"""Formation geometry definitions: PCO and GCO configurations.

Uses the CORRECT natural frequency from the Schweighart-Sedwick equations,
not the approximate n*kappa frequency. This ensures the target formation
oscillates at the same frequency as the actual dynamics.
"""
import numpy as np
from cpfc_simulation.config import (
    FORMATION_RADIUS, N_SATELLITES, MU_EARTH, R_EARTH, ALT_NOMINAL, J2, INC_NOMINAL
)
from cpfc_simulation.dynamics.relative_motion import compute_SS_coefficients


def compute_orbital_params(a=None, inc_deg=None):
    """Compute orbital parameters including SS corrections.

    Returns dict with n, kappa, c, s, a, inc, omega (the correct natural
    frequency of the in-plane SS equations).
    """
    if a is None:
        a = R_EARTH + ALT_NOMINAL
    if inc_deg is None:
        inc_deg = INC_NOMINAL
    inc = np.radians(inc_deg)

    n, kappa, c, s = compute_SS_coefficients(a, inc)

    # Natural frequency of the in-plane SS equations:
    # omega^2 = 4*n^2*kappa^2 - (1+2c)*n^2
    omega2 = 4.0 * n**2 * kappa**2 - (1.0 + 2.0 * c) * n**2
    omega = np.sqrt(max(omega2, 0.0))

    # y/x amplitude ratio for the natural mode
    ratio_yx = 2.0 * n * kappa / omega if omega > 1e-20 else 2.0

    return {
        'n': n, 'kappa': kappa, 'c': c, 's': s,
        'a': a, 'inc': inc,
        'omega': omega, 'ratio_yx': ratio_yx,
    }


def pco_formation_state(t, orbital_params, rho=FORMATION_RADIUS,
                         n_sats=N_SATELLITES, rho_z=None, delta_z=np.pi/2):
    """Generate PCO formation states using the CORRECT SS natural frequency.

    The PCO is defined in terms of the actual natural mode of the SS equations:
        x_i(t) = x_amp * cos(omega*t + phi_i)
        y_i(t) = -(2*n*kappa/omega) * x_amp * sin(omega*t + phi_i)
        z_i(t) = rho_z * sin(omega_z*t + phi_i + delta)

    where x_amp = rho * omega / (2*n*kappa) so that y-amplitude = rho.

    Parameters
    ----------
    t : float
        Time [s].
    orbital_params : dict
        From compute_orbital_params(). Must contain n, kappa, c, omega, ratio_yx.
    rho : float
        Formation radius (y-amplitude) [m].
    n_sats : int
        Number of satellites.
    rho_z : float or None
        Cross-track amplitude [m]. Default rho/2.
    delta_z : float
        Cross-track phase offset [rad].

    Returns
    -------
    states : ndarray, shape (n_sats, 6)
        [x, y, z, xdot, ydot, zdot] for each satellite.
        MATCHES the SS propagator state ordering.
        Satellite 0 = chief (at origin), satellites 1..N-1 = deputies.
    """
    if rho_z is None:
        rho_z = rho / 2

    n = orbital_params['n']
    kappa = orbital_params['kappa']
    omega = orbital_params['omega']
    s = orbital_params['s']
    ratio_yx = orbital_params['ratio_yx']  # = 2*n*kappa/omega

    # x-amplitude such that y-amplitude = rho
    x_amp = rho / ratio_yx  # = rho * omega / (2*n*kappa)

    # Out-of-plane frequency
    omega_z = n * np.sqrt(max(s, 0.0))

    phi = np.linspace(0, 2 * np.pi, n_sats, endpoint=False)
    states = np.zeros((n_sats, 6))

    for i in range(n_sats):
        if i == 0:
            # Chief at origin (no relative offset)
            continue

        phase = omega * t + phi[i]
        phase_z = omega_z * t + phi[i] + delta_z

        # State ordering: [x, y, z, xdot, ydot, zdot]
        # Matches SS propagator (relative_motion.py) convention
        states[i, 0] = x_amp * np.cos(phase)                    # x
        states[i, 1] = -rho * np.sin(phase)                     # y
        states[i, 2] = rho_z * np.sin(phase_z)                  # z
        states[i, 3] = -x_amp * omega * np.sin(phase)           # xdot
        states[i, 4] = -rho * omega * np.cos(phase)              # ydot
        states[i, 5] = rho_z * omega_z * np.cos(phase_z)        # zdot

    return states


def drift_free_initial_conditions(x0, n, kappa):
    """Compute ydot0 for drift-free relative orbit.

    The drift-free condition in SS equations is D = ydot0 + 2*n*kappa*x0 = 0.
    """
    return -2.0 * n * kappa * x0


def string_formation_state(n_sats, separation, along_track=True):
    """Along-track string formation (satellites spaced in y-direction)."""
    states = np.zeros((n_sats, 6))
    for i in range(n_sats):
        if along_track:
            states[i, 2] = i * separation
        else:
            states[i, 0] = i * separation
    return states


def formation_error(current_states, target_states):
    """Compute per-satellite and aggregate formation error.

    State ordering: [x, y, z, xdot, ydot, zdot].
    Position indices: 0 (x), 1 (y), 2 (z).
    """
    errors = current_states - target_states
    per_sat_rms = np.sqrt(np.mean(errors[:, [0, 1, 2]]**2, axis=1))
    total_rms = np.sqrt(np.mean(errors[:, [0, 1, 2]]**2))
    return {'per_satellite_rms': per_sat_rms, 'total_rms': total_rms, 'errors': errors}
