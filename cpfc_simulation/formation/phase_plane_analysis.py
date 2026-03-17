"""Phase plane analysis for formation relative motion with differential drag."""
import numpy as np
from scipy.integrate import solve_ivp
from cpfc_simulation.config import MU_EARTH, R_EARTH, J2, ALT_NOMINAL, INC_NOMINAL


def cw_ellipse(x0, xdot0, y0, ydot0, n, t_array):
    """
    Clohessy-Wiltshire (unperturbed) relative motion ellipse.
    Standard CW solution for comparison baseline.
    """
    c1 = (4*x0 + 2*ydot0/n)
    c2 = xdot0/n
    c3 = -(2*xdot0/n + 3*x0)
    c4 = y0 - 2*xdot0/n
    c5 = -(3*n*x0 + 2*ydot0)

    x = c1 + c3*np.cos(n*t_array) + c2*np.sin(n*t_array)
    y = c4 + c5*t_array + 2*c3*np.sin(n*t_array) - 2*c2*np.cos(n*t_array)
    xdot_t = c3*n*np.sin(n*t_array) + c2*n*np.cos(n*t_array)  # WRONG SIGN - fix

    # Actually use standard CW:
    x_t = (4 - 3*np.cos(n*t_array))*x0 + np.sin(n*t_array)/n * xdot0 + 2*(1-np.cos(n*t_array))/n * ydot0
    xdot_t2 = 3*n*np.sin(n*t_array)*x0 + np.cos(n*t_array)*xdot0 + 2*np.sin(n*t_array)*ydot0
    y_t = 6*(np.sin(n*t_array) - n*t_array)*x0 + y0 - 2*(1-np.cos(n*t_array))/n*xdot0 + (4*np.sin(n*t_array)/n - 3*t_array)*ydot0
    ydot_t = 6*n*(np.cos(n*t_array)-1)*x0 + 2*np.sin(n*t_array)*xdot0 + (4*np.cos(n*t_array) - 3)*ydot0

    return x_t, xdot_t2, y_t, ydot_t


def ss_corrected_ellipse(x0, xdot0, y0, ydot0, n, kappa, c, t_array, dfy=0.0):
    """
    Corrected Schweighart-Sedwick ellipse with Traub 2025 correction.
    """
    nk = n * kappa

    # Integration constants
    C2 = x0
    C3 = xdot0 / nk

    secular_rate = 3*kappa/((1+2*c)*n**2) * dfy
    # y(t) = C0 + (C4 + secular_rate)*t - (2/kappa)*C2*sin(nk*t) + (2/kappa)*C3*cos(nk*t)
    # ydot(t) = (C4 + secular_rate) - (2*nk/kappa)*C2*cos(nk*t) - (2*nk/kappa)*C3*sin(nk*t)
    # At t=0: ydot(0) = (C4 + secular_rate) - (2*nk/kappa)*C2
    # So C4 = ydot0 + (2*nk/kappa)*C2 - secular_rate
    C4 = ydot0 + (2*nk/kappa)*C2 - secular_rate

    # At t=0: y(0) = C0 + (2/kappa)*C3
    # So C0 = y0 - (2/kappa)*C3
    C0 = y0 - (2/kappa) * C3

    x_t = C2*np.cos(nk*t_array) + C3*np.sin(nk*t_array)
    xdot_t = -C2*nk*np.sin(nk*t_array) + C3*nk*np.cos(nk*t_array)
    y_t = C0 + (C4 + secular_rate)*t_array - (2/kappa)*C2*np.sin(nk*t_array) + (2/kappa)*C3*np.cos(nk*t_array)
    ydot_t = (C4 + secular_rate) - (2*nk/kappa)*C2*np.cos(nk*t_array) - (2*nk/kappa)*C3*np.sin(nk*t_array)

    return x_t, xdot_t, y_t, ydot_t


def compute_accessible_ellipses(n, kappa, c, rho, dfy_range, t_array):
    """
    Compute family of relative motion ellipses accessible by different dfy values.
    Returns list of (x_t, y_t) trajectories.
    """
    ellipses = []
    # Start from PCO initial conditions
    x0 = rho / 2
    xdot0 = 0.0
    y0 = 0.0
    ydot0 = -rho * n * kappa

    for dfy in dfy_range:
        x_t, _, y_t, _ = ss_corrected_ellipse(x0, xdot0, y0, ydot0, n, kappa, c, t_array, dfy)
        ellipses.append((x_t, y_t, dfy))

    return ellipses


def compute_reconfiguration_time(current_state, target_state, dfy_max, n, kappa, c):
    """
    Estimate time to reconfigure from current to target formation.
    Based on maximum achievable along-track drift rate.
    """
    dy = target_state[2] - current_state[2]  # along-track difference
    secular_gain = 3*kappa / ((1+2*c) * n**2)
    drift_rate = secular_gain * dfy_max  # [m/s]
    if abs(drift_rate) < 1e-30:
        return np.inf
    return abs(dy) / abs(drift_rate)


def delta_v_equivalent(dfy, T_apply):
    """Compute delta-V equivalent of differential drag applied for T_apply seconds."""
    return abs(dfy) * T_apply
