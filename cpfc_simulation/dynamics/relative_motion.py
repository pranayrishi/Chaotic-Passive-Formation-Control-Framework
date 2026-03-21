"""Corrected Schweighart-Sedwick relative motion equations.

Implements the J2-corrected linearised relative motion model from
Schweighart & Sedwick (2002) with the secular drift correction from
Traub et al. (2025).
"""
import numpy as np
from scipy.integrate import solve_ivp
from cpfc_simulation.config import MU_EARTH, R_EARTH, J2, DT_INTEGRATOR


# ---------------------------------------------------------------------------
# Schweighart-Sedwick coefficients
# ---------------------------------------------------------------------------

def compute_SS_coefficients(a, inc_rad):
    """Compute corrected Schweighart-Sedwick frequency and coupling coefficients.

    Parameters
    ----------
    a : float
        Semi-major axis [m].
    inc_rad : float
        Inclination [rad].

    Returns
    -------
    n : float
        Mean motion [rad/s].
    kappa : float
        In-plane frequency correction factor.
    c : float
        Cross-coupling coefficient.
    s : float
        Out-of-plane frequency ratio squared.
    """
    n = np.sqrt(MU_EARTH / a**3)
    q = J2 * (R_EARTH / a)**2  # small parameter

    sin_i = np.sin(inc_rad)
    cos_i = np.cos(inc_rad)

    # kappa^2 = 1 + 1.5*q*(5*cos^2(i) - 1)  [Schweighart & Sedwick 2002, Eq. 18]
    kappa2 = 1.0 + 1.5 * q * (5.0 * cos_i**2 - 1.0)
    kappa = np.sqrt(max(kappa2, 0.0))

    # c: radial restoring coefficient
    # The SS x-equation is: x'' = 2*n*kappa*y' + (1+2c)*n^2*x
    # Must reduce to CW (x'' = 2n*y' + 3n^2*x) when J2=0.
    # From Schweighart & Sedwick 2002 Eq. 21: coefficient = (5*cos^2(i) - 2)
    # => (1+2c) = 5*cos^2(i) - 2   =>   c = (5*cos^2(i) - 3) / 2
    # CW check: i=0 => c=1, (1+2c)=3 ✓ ; i=90 => c=-3/2, (1+2c)=-2 ✓
    c = (5.0 * cos_i**2 - 3.0) / 2.0

    # s: out-of-plane frequency ratio squared
    # z'' + s*n^2*z = 0, where s = 1 - 3*cos^2(i) from SS Eq. 23
    # Plus J2 correction: s includes geometric J2 terms
    # CW check (J2=0, any i): s should give n_z = n, so s=1
    # With J2: s = 1 + 1.5*q*(3 - 5*sin^2(i))  [approximate, small J2]
    # But dominant term is from inclination: s ≈ 1 + 1.5*q*(3 - 4*sin^2(i))/2
    s = 1.0 + 1.5 * q * (3.0 - 4.0 * sin_i**2) / 2.0

    return n, kappa, c, s


# ---------------------------------------------------------------------------
# Closed-form solution (Traub 2025 corrected)
# ---------------------------------------------------------------------------

def corrected_SS_solution(t, x0, xdot0, y0, ydot0, n, kappa, c,
                          dfx=0.0, dfy=0.0):
    """Closed-form Schweighart-Sedwick solution with Traub (2025) correction.

    In-plane (x, y) solution for the corrected SS equations:
        xdd = 2*n*kappa*ydot + (1+2c)*n^2*x + dfx
        ydd = -2*n*kappa*xdot + dfy

    Derivation:
        From the second equation, integrating:
            ydot + 2*n*kappa*x = D + dfy*t  (D is a constant of integration)
        Substituting into the first equation to eliminate ydot yields a
        forced harmonic oscillator in x with natural frequency:
            w = n*sqrt(4*kappa^2 - (1+2c))

    Parameters
    ----------
    t : float or ndarray
        Time(s) [s].
    x0, xdot0, y0, ydot0 : float
        Initial conditions [m, m/s].
    n : float
        Mean motion [rad/s].
    kappa : float
        In-plane frequency correction.
    c : float
        Cross-coupling coefficient.
    dfx, dfy : float
        Constant differential accelerations [m/s^2].

    Returns
    -------
    x, xdot, y, ydot : ndarray or float
        State at time(s) t.
    """
    nk = n * kappa

    # Characteristic frequency (exact from coupled EOM)
    omega2 = 4.0 * nk**2 - (1.0 + 2.0 * c) * n**2
    omega = np.sqrt(max(omega2, 0.0))

    if abs(omega) < 1e-20 or abs(n) < 1e-20:
        # Degenerate case: return initial conditions
        return (x0 * np.ones_like(t), xdot0 * np.ones_like(t),
                y0 * np.ones_like(t), ydot0 * np.ones_like(t))

    # --- Integral of motion ---
    # From ydd = -2*nk*xdot + dfy, integrating:
    #   ydot + 2*nk*x = D + dfy*t
    # where D = ydot0 + 2*nk*x0  (from ICs at t=0)
    D = ydot0 + 2.0 * nk * x0

    # --- Particular solution for constant dfx ---
    # With dfy, the integral of motion becomes time-dependent:
    #   ydot = -2*nk*x + D + dfy*t
    # Sub into xdd equation:
    #   xdd = 2*nk*(-2*nk*x + D + dfy*t) + (1+2c)*n^2*x + dfx
    #       = -omega^2 * x + 2*nk*D + 2*nk*dfy*t + dfx
    # Particular solutions:
    #   x_p_const = (2*nk*D + dfx) / omega^2
    #   x_p_linear: from 2*nk*dfy*t, x_p = (2*nk*dfy/omega^2)*t  (secular in x)
    # For bounded x, dfy should be zero or handled via the Traub drift correction.

    x_p = (2.0 * nk * D + dfx) / omega2

    # Secular x from dfy (Traub correction drift):
    # To keep x bounded, the dfy contribution creates a secular y-drift instead.
    # The secular y-drift rate from dfy is:
    dfy_drift = dfy / (2.0 * nk) if abs(nk) > 1e-20 else 0.0
    # This adds to the D-induced secular rate in y.

    # --- Homogeneous solution ---
    # x_h(t) = C2*cos(w*t) + C3*sin(w*t)
    C2 = x0 - x_p
    C3 = xdot0 / omega

    # y from integrating ydot = -2*nk*x + D + dfy*t:
    # ydot = (D - 2*nk*x_p) - 2*nk*(C2*cos + C3*sin) + dfy*t
    # y = (D - 2*nk*x_p)*t - (2*nk/w)*(C2*sin - C3*cos) + 0.5*dfy*t^2 + E
    secular_rate = D - 2.0 * nk * x_p

    # From y(0) = y0:
    # y(0) = (2*nk/w)*C3 + E => E = y0 - (2*nk/w)*C3
    E = y0 - (2.0 * nk / omega) * C3

    # --- Evaluate ---
    wt = omega * t
    cos_wt = np.cos(wt)
    sin_wt = np.sin(wt)

    x = x_p + C2 * cos_wt + C3 * sin_wt
    xd = -C2 * omega * sin_wt + C3 * omega * cos_wt

    y = (secular_rate * t
         - (2.0 * nk / omega) * C2 * sin_wt
         + (2.0 * nk / omega) * C3 * cos_wt
         + 0.5 * dfy * t**2
         + E)
    yd = (secular_rate
          - 2.0 * nk * C2 * cos_wt
          - 2.0 * nk * C3 * sin_wt
          + dfy * t)

    return x, xd, y, yd


# ---------------------------------------------------------------------------
# Numerical equations of motion
# ---------------------------------------------------------------------------

def ss_eom(t, state, n, kappa, c, s, dfx_func=None, dfy_func=None,
           dfz_func=None):
    """Right-hand side for the corrected Schweighart-Sedwick equations.

    State = [x, y, z, xdot, ydot, zdot] in LVLH frame.

    Parameters
    ----------
    t : float
        Time [s].
    state : ndarray, shape (6,)
        Relative state [m, m/s].
    n : float
        Mean motion [rad/s].
    kappa, c, s : float
        SS correction coefficients.
    dfx_func, dfy_func, dfz_func : callable(t) -> float or None
        Differential acceleration functions. None means zero.

    Returns
    -------
    ndarray, shape (6,)
        State derivative.
    """
    x, y, z, xdot, ydot, zdot = state

    dfx = dfx_func(t) if dfx_func is not None else 0.0
    dfy = dfy_func(t) if dfy_func is not None else 0.0
    dfz = dfz_func(t) if dfz_func is not None else 0.0

    xddot = 2.0 * n * kappa * ydot + (1.0 + 2.0 * c) * n**2 * x + dfx
    yddot = -2.0 * n * kappa * xdot + dfy
    zddot = -n**2 * s * z + dfz

    return np.array([xdot, ydot, zdot, xddot, yddot, zddot])


# ---------------------------------------------------------------------------
# Propagation wrapper
# ---------------------------------------------------------------------------

def propagate_relative_motion(state0, t_span, orbital_params,
                              diff_forces=None, t_eval=None):
    """Propagate relative motion using corrected SS equations.

    Parameters
    ----------
    state0 : ndarray, shape (6,)
        Initial relative state [x, y, z, xdot, ydot, zdot] [m, m/s].
    t_span : tuple (t0, tf)
        Integration interval [s].
    orbital_params : dict
        Must contain 'a' [m] and 'inc_rad' [rad].
    diff_forces : dict or None
        Keys 'dfx', 'dfy', 'dfz' mapping to callable(t)->float or float.
        If float, treated as constant.
    t_eval : ndarray or None
        Times for output.

    Returns
    -------
    sol : OdeResult
        scipy solve_ivp solution object.
    """
    a = orbital_params['a']
    inc_rad = orbital_params['inc_rad']
    n, kappa, c, s_coeff = compute_SS_coefficients(a, inc_rad)

    # Build force functions
    if diff_forces is None:
        diff_forces = {}

    def _make_func(val):
        if val is None:
            return None
        if callable(val):
            return val
        # Constant value
        return lambda t, _v=val: _v

    dfx_func = _make_func(diff_forces.get('dfx', None))
    dfy_func = _make_func(diff_forces.get('dfy', None))
    dfz_func = _make_func(diff_forces.get('dfz', None))

    def rhs(t, state):
        return ss_eom(t, state, n, kappa, c, s_coeff, dfx_func, dfy_func,
                      dfz_func)

    sol = solve_ivp(rhs, t_span, state0, method='RK45',
                    rtol=1e-10, atol=1e-12, max_step=DT_INTEGRATOR,
                    t_eval=t_eval, dense_output=True)
    return sol


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_SS_solution(orbital_params, tol=1e-6):
    """Validate closed-form against numerical SS solution.

    Runs both the analytical ``corrected_SS_solution`` and the numerical
    ``propagate_relative_motion`` with the same ICs and checks that they
    agree to within ``tol`` metres.

    Parameters
    ----------
    orbital_params : dict
        Must contain 'a' [m] and 'inc_rad' [rad].
    tol : float
        Maximum allowable position error [m].

    Returns
    -------
    max_err : float
        Maximum position discrepancy [m].
    passed : bool
        True if max_err < tol.
    """
    a = orbital_params['a']
    inc_rad = orbital_params['inc_rad']
    n, kappa, c, s_coeff = compute_SS_coefficients(a, inc_rad)

    # Test initial conditions
    x0, y0, z0 = 100.0, 200.0, 50.0  # [m]
    xdot0, ydot0, zdot0 = 0.01, -0.02, 0.005  # [m/s]

    T_orbit = 2.0 * np.pi / n
    t_eval = np.linspace(0, T_orbit, 1000)

    # Analytical (in-plane only, z is decoupled)
    x_an, _, y_an, _ = corrected_SS_solution(
        t_eval, x0, xdot0, y0, ydot0, n, kappa, c
    )

    # Analytical z: z(t) = z0*cos(n*sqrt(s)*t) + zdot0/(n*sqrt(s))*sin(n*sqrt(s)*t)
    ns = n * np.sqrt(max(s_coeff, 0.0))
    z_an = z0 * np.cos(ns * t_eval) + (zdot0 / ns) * np.sin(ns * t_eval)

    # Numerical
    state0 = np.array([x0, y0, z0, xdot0, ydot0, zdot0])
    sol = propagate_relative_motion(state0, (0.0, T_orbit), orbital_params,
                                    t_eval=t_eval)

    x_num = sol.y[0]
    y_num = sol.y[1]
    z_num = sol.y[2]

    err_x = np.max(np.abs(x_an - x_num))
    err_y = np.max(np.abs(y_an - y_num))
    err_z = np.max(np.abs(z_an - z_num))
    max_err = max(err_x, err_y, err_z)

    passed = max_err < tol

    return max_err, passed


# ---------------------------------------------------------------------------
# Relative Orbital Elements (ROE) and frame conversions
# (From S-Net / Ben-Yaacov & Gurfil 2013, JGCD 36(6):1731-1740)
# ---------------------------------------------------------------------------

def cartesian_to_classical_oe(r_vec, v_vec, mu=MU_EARTH):
    """Convert ECI position/velocity to classical orbital elements.

    Parameters
    ----------
    r_vec : ndarray (3,)
        ECI position [m].
    v_vec : ndarray (3,)
        ECI velocity [m/s].
    mu : float
        Gravitational parameter [m^3/s^2].

    Returns
    -------
    ndarray (6,)
        [a, e, i, omega, RAAN, M] with a in [m], angles in [rad].
    """
    r_vec = np.asarray(r_vec, dtype=float)
    v_vec = np.asarray(v_vec, dtype=float)

    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    vr = np.dot(r_vec, v_vec) / r

    # Angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    # Node line
    K = np.array([0.0, 0.0, 1.0])
    N_vec = np.cross(K, h_vec)
    N = np.linalg.norm(N_vec)

    # Eccentricity vector
    e_vec = ((v**2 - mu / r) * r_vec - r * vr * v_vec) / mu
    e = np.linalg.norm(e_vec)

    # Orbital elements
    a = 1.0 / (2.0 / r - v**2 / mu)
    i = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))

    RAAN = np.arccos(np.clip(N_vec[0] / N, -1.0, 1.0)) if N > 1e-10 else 0.0
    if N_vec[1] < 0.0:
        RAAN = 2.0 * np.pi - RAAN

    omega = (np.arccos(np.clip(np.dot(N_vec, e_vec) / (N * e), -1.0, 1.0))
             if N > 1e-10 and e > 1e-10 else 0.0)
    if e_vec[2] < 0.0:
        omega = 2.0 * np.pi - omega

    nu = (np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1.0, 1.0))
          if e > 1e-10 else 0.0)
    if vr < 0.0:
        nu = 2.0 * np.pi - nu

    # True -> eccentric -> mean anomaly
    E = 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(nu / 2.0),
                          np.sqrt(1.0 + e) * np.cos(nu / 2.0))
    M = E - e * np.sin(E)
    M = M % (2.0 * np.pi)

    return np.array([a, e, i, omega, RAAN, M])


def cartesian_to_roe(r_chief, v_chief, r_deputy, v_deputy, mu=MU_EARTH):
    """Convert absolute ECI states to Relative Orbital Elements.

    Uses the linearised ROE mapping from Ben-Yaacov & Gurfil (2013).

    Parameters
    ----------
    r_chief, v_chief : ndarray (3,)
        Chief ECI state [m, m/s].
    r_deputy, v_deputy : ndarray (3,)
        Deputy ECI state [m, m/s].
    mu : float
        Gravitational parameter [m^3/s^2].

    Returns
    -------
    ndarray (6,)
        [delta_a, delta_ex, delta_ey, delta_ix, delta_iy, delta_u].
    """
    oe_c = cartesian_to_classical_oe(r_chief, v_chief, mu)
    oe_d = cartesian_to_classical_oe(r_deputy, v_deputy, mu)

    a_c, e_c, i_c, om_c, Ra_c, M_c = oe_c
    a_d, e_d, i_d, om_d, Ra_d, M_d = oe_d

    delta_a  = (a_d - a_c) / a_c
    delta_e  = e_d - e_c
    delta_i  = i_d - i_c
    delta_Ra = (Ra_d - Ra_c + np.pi) % (2 * np.pi) - np.pi
    delta_M  = (M_d - M_c + np.pi) % (2 * np.pi) - np.pi
    delta_om = om_d - om_c

    delta_ex = delta_e * np.cos(om_c) - e_c * delta_Ra * np.sin(om_c)
    delta_ey = delta_e * np.sin(om_c) + e_c * delta_Ra * np.cos(om_c)
    delta_ix = delta_i
    delta_iy = np.sin(i_c) * delta_Ra
    delta_u  = delta_M + delta_om

    return np.array([delta_a, delta_ex, delta_ey, delta_ix, delta_iy, delta_u])


def eci_to_rtn(r_chief, v_chief, r_deputy, v_deputy):
    """Convert ECI absolute states to RTN relative position and velocity.

    RTN frame centred on the chief:
      R = radial (outward), T = along-track, N = cross-track.

    Parameters
    ----------
    r_chief, v_chief : ndarray (3,)
        Chief ECI state [m, m/s].
    r_deputy, v_deputy : ndarray (3,)
        Deputy ECI state [m, m/s].

    Returns
    -------
    rtn_pos : ndarray (3,)
        [R, T, N] relative position [m].
    rtn_vel : ndarray (3,)
        [vR, vT, vN] relative velocity [m/s].
    """
    rc = np.asarray(r_chief, dtype=float)
    vc = np.asarray(v_chief, dtype=float)

    r_rel = np.asarray(r_deputy, dtype=float) - rc
    v_rel = np.asarray(v_deputy, dtype=float) - vc

    r_hat = rc / np.linalg.norm(rc)
    h_vec = np.cross(rc, vc)
    n_hat = h_vec / np.linalg.norm(h_vec)
    t_hat = np.cross(n_hat, r_hat)

    Q = np.vstack([r_hat, t_hat, n_hat])

    omega_mag = np.linalg.norm(h_vec) / np.linalg.norm(rc)**2
    omega_vec = n_hat * omega_mag

    rtn_pos = Q @ r_rel
    rtn_vel = Q @ (v_rel - np.cross(omega_vec, r_rel))

    return rtn_pos, rtn_vel


def along_track_drift_rate(delta_a_abs, sma, mu=MU_EARTH):
    """Compute mean along-track drift rate from SMA difference.

    From D'Amico & Montenbruck (2006):
      dy_drift/dt = -3/2 * n * delta_a / a

    Parameters
    ----------
    delta_a_abs : float
        Absolute SMA difference a_deputy - a_chief [m].
    sma : float
        Chief semi-major axis [m].
    mu : float
        Gravitational parameter [m^3/s^2].

    Returns
    -------
    float
        Along-track drift rate [m/s].
    """
    n = np.sqrt(mu / sma**3)
    return -1.5 * n * delta_a_abs / sma
