"""Library of chaotic attractors for attitude dynamics."""
import numpy as np
from scipy.integrate import solve_ivp
from cpfc_simulation.config import IXX, IYY, IZZ


def lorenz_system(t, state, sigma=10.0, rho=28.0, beta=8/3):
    """Classic Lorenz attractor for comparison/validation."""
    x, y, z = state
    return [sigma*(y - x), x*(rho - z) - y, x*y - beta*z]


def rossler_system(t, state, a=0.2, b=0.2, c=5.7):
    """Rossler attractor for comparison."""
    x, y, z = state
    return [-y - z, x + a*y, b + z*(x - c)]


def doroshin_attractor_1(t, state, Ixx=IXX, Iyy=IYY, Izz=IZZ,
                          tau_ext_x=0.0, tau_ext_y=0.0, tau_ext_z=0.0):
    """
    Doroshin & Elisov (2024) Attractor Type 1: Tumbling-spinning regime.
    Euler equations with constant external torques that produce chaos.
    """
    ox, oy, oz = state
    dox = ((Iyy - Izz) * oy * oz + tau_ext_x) / Ixx
    doy = ((Izz - Ixx) * oz * ox + tau_ext_y) / Iyy
    doz = ((Ixx - Iyy) * ox * oy + tau_ext_z) / Izz
    return [dox, doy, doz]


def doroshin_attractor_2(t, state, Ixx=IXX, Iyy=IYY, Izz=IZZ,
                          A=1e-4, Omega=0.05):
    """
    Doroshin Attractor Type 2: Periodic forcing in x-axis.
    tau_ext_x = A*sin(Omega*t), tau_ext_y = 0, tau_ext_z = 0
    """
    ox, oy, oz = state
    tau_x = A * np.sin(Omega * t)
    dox = ((Iyy - Izz) * oy * oz + tau_x) / Ixx
    doy = ((Izz - Ixx) * oz * ox) / Iyy
    doz = ((Ixx - Iyy) * ox * oy) / Izz
    return [dox, doy, doz]


def doroshin_attractor_3(t, state, Ixx=IXX, Iyy=IYY, Izz=IZZ,
                          A1=1e-4, Omega1=0.05, A2=5e-5, Omega2=None):
    """
    Doroshin Attractor Type 3: Quasi-periodic forcing in x and y axes.
    tau_ext_x = A1*sin(Omega1*t), tau_ext_y = A2*sin(Omega2*t)
    where Omega2/Omega1 is irrational (golden ratio used).
    """
    if Omega2 is None:
        Omega2 = Omega1 * (1 + np.sqrt(5)) / 2  # golden ratio * Omega1

    ox, oy, oz = state
    tau_x = A1 * np.sin(Omega1 * t)
    tau_y = A2 * np.sin(Omega2 * t)
    dox = ((Iyy - Izz) * oy * oz + tau_x) / Ixx
    doy = ((Izz - Ixx) * oz * ox + tau_y) / Iyy
    doz = ((Ixx - Iyy) * ox * oy) / Izz
    return [dox, doy, doz]


def doroshin_attractor_4(t, state, Ixx=IXX, Iyy=IYY, Izz=IZZ,
                          eps=0.1, Omega_p=0.03):
    """
    Doroshin Attractor Type 4: Parametric excitation (time-varying inertia from panel deployment).
    Ixx_eff(t) = Ixx*(1 + eps*sin(Omega_p*t))
    Euler equations with time-varying moment of inertia.
    """
    ox, oy, oz = state
    Ixx_eff = Ixx * (1.0 + eps * np.sin(Omega_p * t))
    # Time derivative of Ixx_eff for the full equation:
    # d(Ixx_eff * ox)/dt = (Iyy - Izz)*oy*oz
    # Ixx_eff * dox + dIxx_eff/dt * ox = (Iyy - Izz)*oy*oz
    dIxx_dt = Ixx * eps * Omega_p * np.cos(Omega_p * t)

    dox = ((Iyy - Izz) * oy * oz - dIxx_dt * ox) / Ixx_eff
    doy = ((Izz - Ixx_eff) * oz * ox) / Iyy
    doz = ((Ixx_eff - Iyy) * ox * oy) / Izz
    return [dox, doy, doz]


def doroshin_attractor_5(t, state, Ixx=IXX, Iyy=IYY, Izz=IZZ,
                          n_orb=0.00113, M_gg_coeff=None, M_aero_coeff=1e-5):
    """
    Doroshin Attractor Type 5: Combined gravity-gradient + aerodynamic torque.
    The CubeSat-specific attractor.

    State = [ox, oy, oz, theta, alpha]
    where theta = pitch angle, alpha = angle of attack.

    tau_ext_y = -M_gg*sin(2*theta) - M_aero*sin(alpha)
    with theta_dot = oy (pitch rate) and alpha_dot ~ oy - n_orb.
    """
    ox, oy, oz, theta, alpha = state

    if M_gg_coeff is None:
        M_gg_coeff = 1.5 * n_orb**2 * abs(Ixx - Izz)

    tau_y = -M_gg_coeff * np.sin(2.0 * theta) - M_aero_coeff * np.sin(alpha)

    dox = ((Iyy - Izz) * oy * oz) / Ixx
    doy = ((Izz - Ixx) * oz * ox + tau_y) / Iyy
    doz = ((Ixx - Iyy) * ox * oy) / Izz
    dtheta = oy
    dalpha = oy - n_orb

    return [dox, doy, doz, dtheta, dalpha]


def generate_attractor(attractor_func, state0, T, dt, **kwargs):
    """
    Integrate an attractor system and return the trajectory.

    Parameters
    ----------
    attractor_func : callable
        EOM function f(t, state, **kwargs).
    state0 : array-like
        Initial condition.
    T : float
        Total integration time.
    dt : float
        Desired output time step.
    **kwargs :
        Additional keyword arguments passed to attractor_func.

    Returns
    -------
    t_array : ndarray, shape (N,)
    trajectory : ndarray, shape (N, n_dim)
    """
    t_eval = np.arange(0, T, dt)

    def eom(t, state):
        return attractor_func(t, state, **kwargs)

    sol = solve_ivp(eom, [0, T], state0, method='RK45',
                    t_eval=t_eval, rtol=1e-10, atol=1e-12, max_step=dt)
    return sol.t, sol.y.T


def classify_attractor(trajectory, dt=1.0):
    """
    Classify trajectory as fixed point, limit cycle, quasi-periodic, or chaotic
    using the maximum Lyapunov exponent computed via nearest-neighbor divergence
    (Rosenstein et al. 1993 approach, simplified).

    Parameters
    ----------
    trajectory : ndarray, shape (N, n_dim)
        Time series of the system.
    dt : float
        Sampling time step.

    Returns
    -------
    classification : str
        One of 'fixed_point', 'limit_cycle', 'quasi_periodic', 'chaotic'.
    mle_estimate : float
        Estimated maximum Lyapunov exponent.
    """
    N, n_dim = trajectory.shape

    if N < 100:
        return 'insufficient_data', 0.0

    # Check if trajectory converges to a point
    std_last = np.std(trajectory[-N//4:], axis=0)
    if np.all(std_last < 1e-8):
        return 'fixed_point', -np.inf

    # Estimate MLE via average divergence of nearby trajectories
    # (simplified Rosenstein method)
    mean_traj = np.mean(trajectory, axis=0)
    traj_centered = trajectory - mean_traj

    # Find nearest neighbors (excluding temporal neighbors)
    min_temporal_sep = max(10, N // 50)
    n_reference = min(200, N // 2)
    ref_indices = np.linspace(N // 10, N - N // 10, n_reference, dtype=int)

    divergences = []
    for idx in ref_indices:
        point = trajectory[idx]
        # Compute distances to all other points
        dists = np.linalg.norm(trajectory - point, axis=1)
        # Exclude temporal neighbors
        dists[max(0, idx - min_temporal_sep):min(N, idx + min_temporal_sep)] = np.inf
        nn_idx = np.argmin(dists)

        if dists[nn_idx] < 1e-30:
            continue

        # Track divergence for a short time
        horizon = min(min_temporal_sep * 5, N - max(idx, nn_idx) - 1)
        if horizon < 5:
            continue

        d0 = dists[nn_idx]
        for k in range(1, horizon):
            if idx + k >= N or nn_idx + k >= N:
                break
            dk = np.linalg.norm(trajectory[idx + k] - trajectory[nn_idx + k])
            if dk > 0 and d0 > 0:
                divergences.append(np.log(dk / d0) / (k * dt))

    if len(divergences) == 0:
        return 'limit_cycle', 0.0

    mle = np.median(divergences)

    if mle > 0.01:
        return 'chaotic', mle
    elif mle > -0.001:
        return 'quasi_periodic', mle
    elif mle > -0.1:
        return 'limit_cycle', mle
    else:
        return 'fixed_point', mle


def time_averaged_cross_section(trajectory, aero_model=None,
                                 Cd_base=2.2, A_ref=0.01,
                                 panel_area=0.06, panel_normal_body=None):
    """
    Compute time-averaged <Cd*A> over an attractor trajectory.
    THIS IS THE KEY LINK TO FORMATION CONTROL.

    The effective cross-section depends on the attitude (which varies over the attractor),
    so averaging over the chaotic trajectory gives the effective drag area for
    differential drag computation.

    Parameters
    ----------
    trajectory : ndarray, shape (N, n_dim)
        Attitude trajectory (at minimum columns for [ox, oy, oz] or
        includes [theta, alpha] for Type 5).
    aero_model : callable or None
        If provided, aero_model(attitude_state) -> Cd*A.
        If None, use a geometric model based on the attitude angles.
    Cd_base : float
        Base drag coefficient for a flat plate.
    A_ref : float
        Reference body area (stowed).
    panel_area : float
        Deployed panel area (both sides).
    panel_normal_body : ndarray or None
        Panel normal direction in body frame. Default: [0, 0, 1].

    Returns
    -------
    CdA_mean : float
        Time-averaged Cd*A product.
    CdA_std : float
        Standard deviation (measure of variability).
    CdA_series : ndarray
        Cd*A at each trajectory point.
    """
    N = trajectory.shape[0]
    n_dim = trajectory.shape[1]

    if panel_normal_body is None:
        panel_normal_body = np.array([0.0, 0.0, 1.0])

    CdA_series = np.zeros(N)

    if aero_model is not None:
        for i in range(N):
            CdA_series[i] = aero_model(trajectory[i])
    else:
        # Geometric model: effective area depends on attitude
        # For 3-state (ox, oy, oz): estimate attitude from angular velocity history
        # For 5-state (ox, oy, oz, theta, alpha): use theta and alpha directly
        if n_dim >= 5:
            # Type 5 attractor: use theta (col 3) and alpha (col 4)
            theta = trajectory[:, 3]
            alpha = trajectory[:, 4]
            # Projected body area: body sees ram direction rotated by theta
            # Effective area = A_ref*|cos(theta)| + A_side*|sin(theta)| + panel contribution
            A_side = A_ref * 3.405  # approximate side area ratio for 3U
            body_area = A_ref * np.abs(np.cos(theta)) + A_side * np.abs(np.sin(theta))
            # Panel contribution depends on angle of attack
            panel_proj = panel_area * np.abs(np.cos(alpha))
            CdA_series = Cd_base * (body_area + panel_proj)
        else:
            # For 3-state: use angular velocity magnitude as proxy for tumbling
            omega_mag = np.linalg.norm(trajectory[:, :3], axis=1)
            # Tumbling satellite: average over all orientations
            # More tumbling -> closer to spherical average
            # Spherical average of projected area for a rectangular prism:
            # <A> = (A_x + A_y + A_z) / 4  (for convex body)
            A_x = 0.1 * 0.3405   # L_Y * L_Z
            A_y = 0.1 * 0.3405   # L_X * L_Z
            A_z = 0.1 * 0.1      # L_X * L_Y (= A_ref)
            A_sphere_avg = (A_x + A_y + A_z) / 4.0

            # Interpolate between ram-facing (low tumble) and spherical average (high tumble)
            omega_scale = omega_mag / (omega_mag.max() + 1e-30)
            CdA_series = Cd_base * (A_ref * (1 - omega_scale) + A_sphere_avg * omega_scale
                                    + panel_area * 0.5)  # panels partially deployed average

    CdA_mean = np.mean(CdA_series)
    CdA_std = np.std(CdA_series)

    return CdA_mean, CdA_std, CdA_series
