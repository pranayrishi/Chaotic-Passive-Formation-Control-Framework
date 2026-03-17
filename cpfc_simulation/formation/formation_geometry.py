"""Formation geometry definitions: PCO and GCO configurations."""
import numpy as np
from cpfc_simulation.config import (
    FORMATION_RADIUS, N_SATELLITES, MU_EARTH, R_EARTH, ALT_NOMINAL, J2, INC_NOMINAL
)


def compute_orbital_params(a=None, inc_deg=None):
    """Compute orbital parameters including SS corrections."""
    if a is None:
        a = R_EARTH + ALT_NOMINAL
    if inc_deg is None:
        inc_deg = INC_NOMINAL
    inc = np.radians(inc_deg)
    n = np.sqrt(MU_EARTH / a**3)
    kappa2 = 1 + 1.5 * J2 * (R_EARTH/a)**2 * (5*np.cos(inc)**2 - 1)
    kappa = np.sqrt(abs(kappa2)) * np.sign(kappa2) if kappa2 != 0 else 1.0
    c = 1.5 * J2 * (R_EARTH/a)**2 * (1 - 1.25*np.sin(inc)**2)
    s = 1 + 1.5 * J2 * (R_EARTH/a)**2 * (3 - 4*np.sin(inc)**2) / 2
    return {'n': n, 'kappa': kappa, 'c': c, 's': s, 'a': a, 'inc': inc}


def pco_formation_state(t, n, kappa, rho=FORMATION_RADIUS, n_sats=N_SATELLITES,
                         rho_z=None, delta_z=np.pi/2):
    """
    Generate PCO formation states for all satellites at time t.
    x_i(t) = (rho/2)*cos(n*kappa*t + phi_i)
    y_i(t) = -rho*sin(n*kappa*t + phi_i)
    z_i(t) = rho_z*sin(n*kappa*t + phi_i + delta)

    Returns: array of shape (n_sats, 6) — [x, xdot, y, ydot, z, zdot] for each sat
    """
    if rho_z is None:
        rho_z = rho / 2

    nk = n * kappa
    phi = np.linspace(0, 2*np.pi, n_sats, endpoint=False)  # equal spacing
    states = np.zeros((n_sats, 6))

    for i in range(n_sats):
        phase = nk * t + phi[i]
        phase_z = phase + delta_z

        states[i, 0] = (rho/2) * np.cos(phase)           # x
        states[i, 1] = -(rho/2) * nk * np.sin(phase)     # xdot
        states[i, 2] = -rho * np.sin(phase)                # y
        states[i, 3] = -rho * nk * np.cos(phase)           # ydot
        states[i, 4] = rho_z * np.sin(phase_z)             # z
        states[i, 5] = rho_z * nk * np.cos(phase_z)        # zdot

    return states


def gco_drift_free_condition(x0, n):
    """Compute ydot0 for drift-free GCO: ydot0 = -2*n*x0."""
    return -2 * n * x0


def string_formation_state(n_sats, separation, along_track=True):
    """Along-track string formation (satellites spaced in y-direction)."""
    states = np.zeros((n_sats, 6))
    for i in range(n_sats):
        if along_track:
            states[i, 2] = i * separation  # y-spacing
        else:
            states[i, 0] = i * separation  # x-spacing (radial)
    return states


def formation_error(current_states, target_states):
    """Compute per-satellite and aggregate formation error."""
    errors = current_states - target_states
    per_sat_rms = np.sqrt(np.mean(errors[:, [0,2,4]]**2, axis=1))
    total_rms = np.sqrt(np.mean(errors[:, [0,2,4]]**2))
    return {'per_satellite_rms': per_sat_rms, 'total_rms': total_rms, 'errors': errors}
