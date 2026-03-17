"""Lyapunov exponent and FTLE field computation."""
import numpy as np
from scipy.integrate import solve_ivp


def compute_MLE_wolf(system_eom, state0, dt, T_total, renorm_steps=100):
    """
    Wolf et al. (1985) algorithm for maximum Lyapunov exponent.

    Algorithm:
    1. Start with reference trajectory and nearby trajectory (displaced by d0)
    2. Integrate both for renorm_steps * dt time
    3. Measure separation d1
    4. MLE contribution = ln(d1/d0) / (renorm_steps * dt)
    5. Renormalize: reset nearby trajectory to distance d0 along current separation direction
    6. Repeat until T_total
    7. MLE = average of all contributions

    Returns: mle_value, mle_time_series (running average)
    """
    d0 = 1e-8  # initial separation
    n_dim = len(state0)

    # Perturbation vector (along first component)
    delta = np.zeros(n_dim)
    delta[0] = d0

    state_ref = np.array(state0, dtype=float)
    state_pert = state_ref + delta

    renorm_time = renorm_steps * dt
    n_renorms = int(T_total / renorm_time)

    lyap_sum = 0.0
    mle_series = []
    t_current = 0.0

    for i in range(n_renorms):
        t_span = (t_current, t_current + renorm_time)

        sol_ref = solve_ivp(system_eom, t_span, state_ref, method='RK45',
                            rtol=1e-10, atol=1e-12)
        sol_pert = solve_ivp(system_eom, t_span, state_pert, method='RK45',
                             rtol=1e-10, atol=1e-12)

        state_ref = sol_ref.y[:, -1]
        state_pert = sol_pert.y[:, -1]

        # Measure separation
        diff = state_pert - state_ref
        d1 = np.linalg.norm(diff)

        if d1 > 0:
            lyap_sum += np.log(d1 / d0)
            # Renormalize
            state_pert = state_ref + diff * (d0 / d1)

        t_current += renorm_time
        mle_series.append(lyap_sum / t_current)

    mle = lyap_sum / T_total if T_total > 0 else 0.0
    return mle, np.array(mle_series)


def compute_short_time_lyapunov(system_eom, state, dt, T_window=100.0):
    """
    Short-time (finite-time) Lyapunov exponent for real-time chaos monitoring.
    Used by CAPR law to check if system is in chaotic regime.
    """
    d0 = 1e-9
    delta = np.zeros(len(state))
    delta[0] = d0

    state_pert = state + delta
    sol_ref = solve_ivp(system_eom, [0, T_window], state, method='RK45',
                        rtol=1e-10, atol=1e-12)
    sol_pert = solve_ivp(system_eom, [0, T_window], state_pert, method='RK45',
                         rtol=1e-10, atol=1e-12)

    d1 = np.linalg.norm(sol_pert.y[:, -1] - sol_ref.y[:, -1])
    if d1 > 0 and d0 > 0:
        return np.log(d1 / d0) / T_window
    return 0.0


def compute_FTLE_field(system_eom, grid_centers, grid_deltas, T_integration, n_dim=4):
    """
    Compute Finite-Time Lyapunov Exponent field over a grid.

    grid_centers: array of shape (N_points, n_dim) -- center of each grid cell
    grid_deltas: perturbation size for each dimension
    T_integration: integration time

    Returns: FTLE values at each grid point (ridges = LCS = manifolds)

    Algorithm:
    For each grid point x0:
    1. Create 2*n_dim perturbed ICs: x0 +/- delta_i * e_i for each dimension i
    2. Integrate all to time T
    3. Compute deformation gradient F = d(phi_T(x))/dx using finite differences
    4. Compute max singular value sigma_max of F
    5. FTLE = (1/T) * ln(sigma_max)
    """
    N = len(grid_centers)
    ftle = np.zeros(N)

    for k in range(N):
        x0 = grid_centers[k]
        # Deformation gradient via finite differences
        F = np.zeros((n_dim, n_dim))

        for i in range(n_dim):
            # Forward perturbation
            x_plus = x0.copy()
            x_plus[i] += grid_deltas[i]
            sol_plus = solve_ivp(system_eom, [0, T_integration], x_plus,
                                 method='RK45', rtol=1e-9, atol=1e-11)

            # Backward perturbation
            x_minus = x0.copy()
            x_minus[i] -= grid_deltas[i]
            sol_minus = solve_ivp(system_eom, [0, T_integration], x_minus,
                                  method='RK45', rtol=1e-9, atol=1e-11)

            # Central difference for column i of F
            F[:, i] = (sol_plus.y[:n_dim, -1] - sol_minus.y[:n_dim, -1]) / (2 * grid_deltas[i])

        # Cauchy-Green strain tensor
        C = F.T @ F
        eigenvalues = np.linalg.eigvalsh(C)
        sigma_max = np.sqrt(max(eigenvalues.max(), 1e-30))
        ftle[k] = np.log(sigma_max) / abs(T_integration) if T_integration != 0 else 0.0

    return ftle
