"""Melnikov integral computation for chaos prediction in attitude dynamics."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from cpfc_simulation.config import (
    MU_EARTH, R_EARTH, J2, IXX, IYY, IZZ, ALT_NOMINAL, MASS_SAT
)


def pitch_potential(theta, n, Ixx, Izz, Iyy, M_aero_coeff=0.0):
    """
    Potential energy for pitch dynamics.
    V(theta) = -(3n^2/2)*(Ixx-Izz)/Iyy * cos^2(theta) - M_aero_coeff*cos(theta)
    """
    return -(3*n**2/2) * (Ixx - Izz)/Iyy * np.cos(theta)**2 - M_aero_coeff * np.cos(theta)


def find_unstable_equilibrium(n, Ixx, Izz, Iyy, M_aero_coeff=0.0):
    """
    Find the unstable equilibrium angle theta_u where dV/dtheta=0 and d2V/dtheta2<0.
    dV/dtheta = 3n^2*(Ixx-Izz)/Iyy * sin(theta)*cos(theta) + M_aero_coeff*sin(theta)
    """
    def dV(theta):
        return 3*n**2*(Ixx-Izz)/Iyy * np.sin(theta)*np.cos(theta) + M_aero_coeff*np.sin(theta)

    # For gravity-gradient only (M_aero=0), unstable eq is at theta = pi/2
    # With aerodynamic, it shifts
    if abs(M_aero_coeff) < 1e-15:
        return np.pi / 2
    try:
        theta_u = brentq(dV, 0.1, np.pi - 0.1)
        return theta_u
    except ValueError:
        return np.pi / 2


def compute_heteroclinic_orbit(n, Ixx, Izz, Iyy, M_aero_coeff=0.0, n_points=5000, T_half=500.0):
    """
    Numerically compute the heteroclinic orbit of the unperturbed pitch dynamics.
    Uses backward/forward integration from the unstable equilibrium along the separatrix.

    Returns: tau_array, theta_h, omega_h (arrays along the heteroclinic orbit)
    """
    theta_u = find_unstable_equilibrium(n, Ixx, Izz, Iyy, M_aero_coeff)

    # Energy at unstable equilibrium
    E_sep = pitch_potential(theta_u, n, Ixx, Izz, Iyy, M_aero_coeff)

    # Pitch dynamics: theta_dot = omega, omega_dot = -dV/dtheta
    def pitch_eom(t, state):
        theta, omega = state
        dVdtheta = 3*n**2*(Ixx-Izz)/Iyy * np.sin(theta)*np.cos(theta) + M_aero_coeff*np.sin(theta)
        return [omega, -dVdtheta]

    # Small perturbation along unstable manifold direction
    # Linearize around theta_u: d2V/dtheta2 at theta_u gives the eigenvalue
    d2V = 3*n**2*(Ixx-Izz)/Iyy * (np.cos(theta_u)**2 - np.sin(theta_u)**2) + M_aero_coeff*np.cos(theta_u)
    lam = np.sqrt(abs(d2V))  # eigenvalue magnitude

    eps = 1e-6
    # Forward integration (moving away from theta_u towards -theta_u or 0)
    state0_fwd = [theta_u - eps, -lam * eps]
    sol_fwd = solve_ivp(pitch_eom, [0, T_half], state0_fwd, method='RK45',
                        rtol=1e-12, atol=1e-14, max_step=0.1,
                        t_eval=np.linspace(0, T_half, n_points//2))

    # Backward integration
    state0_bwd = [theta_u + eps, lam * eps]
    sol_bwd = solve_ivp(pitch_eom, [0, -T_half], state0_bwd, method='RK45',
                        rtol=1e-12, atol=1e-14, max_step=0.1,
                        t_eval=np.linspace(0, -T_half, n_points//2))

    # Combine: backward (reversed) + forward
    tau = np.concatenate([sol_bwd.t[::-1], sol_fwd.t[1:]])
    theta_h = np.concatenate([sol_bwd.y[0][::-1], sol_fwd.y[0][1:]])
    omega_h = np.concatenate([sol_bwd.y[1][::-1], sol_fwd.y[1][1:]])

    return tau, theta_h, omega_h


def melnikov_integral(tau_array, omega_h, Omega):
    """
    Numerical Melnikov integral via trapezoidal rule.
    M = |integral omega_h(tau) * exp(i*Omega*tau) dtau|
    """
    integrand = omega_h * np.exp(1j * Omega * tau_array)
    M_complex = np.trapz(integrand, tau_array)
    return np.abs(M_complex)


def melnikov_chaos_boundary(n, Ixx, Izz, Iyy, Omega_range, M_aero_coeff=0.0):
    """
    Compute Melnikov integral over a range of perturbation frequencies.
    Returns: Omega_range, M_values (|M| for each Omega)
    Chaos exists where M > 0 (simple zeros of M(t0)).
    """
    tau, theta_h, omega_h = compute_heteroclinic_orbit(n, Ixx, Izz, Iyy, M_aero_coeff)
    M_values = np.array([melnikov_integral(tau, omega_h, Om) for Om in Omega_range])
    return Omega_range, M_values


def optimal_switching_frequency(n, Ixx, Izz, Iyy, M_aero_coeff=0.0,
                                Omega_range=None):
    """Find the switching frequency that maximizes the Melnikov integral (strongest chaos)."""
    if Omega_range is None:
        # Scan around natural pitch frequency
        n_pitch = np.sqrt(abs(3*n**2*(Ixx-Izz)/Iyy))
        Omega_range = np.linspace(0.1*n_pitch, 5*n_pitch, 200)

    _, M_vals = melnikov_chaos_boundary(n, Ixx, Izz, Iyy, Omega_range, M_aero_coeff)
    idx_max = np.argmax(M_vals)
    return Omega_range[idx_max], M_vals[idx_max]
