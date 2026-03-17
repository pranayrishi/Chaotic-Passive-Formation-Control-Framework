"""Euler attitude dynamics for rigid body with external torques.

Implements full 3-axis Euler equations with gravity gradient torque,
kinematic equations (3-2-1 sequence), and a simplified pitch-only
mode for chaos studies.
"""
import numpy as np
from scipy.integrate import solve_ivp
from cpfc_simulation.config import MU_EARTH, IXX, IYY, IZZ, DT_INTEGRATOR


# ---------------------------------------------------------------------------
# Kinematic equations (3-2-1 Euler angle sequence)
# ---------------------------------------------------------------------------

def kinematic_equations(omega, theta, phi):
    """Euler angle rates from body angular velocity (3-2-1 sequence).

    Parameters
    ----------
    omega : array-like, shape (3,)
        Body angular velocity [omega_x, omega_y, omega_z] [rad/s].
    theta : float
        Pitch angle [rad].
    phi : float
        Roll angle [rad].

    Returns
    -------
    theta_dot : float
        Pitch rate [rad/s].
    phi_dot : float
        Roll rate [rad/s].
    psi_dot : float
        Yaw rate [rad/s].
    """
    omega_x, omega_y, omega_z = omega

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    tan_theta = np.tan(theta)
    cos_theta = np.cos(theta)

    theta_dot = omega_y * cos_phi - omega_z * sin_phi

    phi_dot = omega_x + (omega_y * sin_phi + omega_z * cos_phi) * tan_theta

    if abs(cos_theta) < 1e-12:
        # Near gimbal lock; degrade gracefully
        psi_dot = 0.0
    else:
        psi_dot = (omega_y * sin_phi + omega_z * cos_phi) / cos_theta

    return theta_dot, phi_dot, psi_dot


# ---------------------------------------------------------------------------
# Gravity gradient torque
# ---------------------------------------------------------------------------

def gravity_gradient_torque(r_body_unit, Ixx, Iyy, Izz, R, mu=MU_EARTH):
    """Gravity gradient torque in principal body axes.

    Parameters
    ----------
    r_body_unit : array-like, shape (3,)
        Nadir unit vector in body frame [r_hat_x, r_hat_y, r_hat_z].
    Ixx, Iyy, Izz : float
        Principal moments of inertia [kg*m^2].
    R : float
        Orbital radius (distance from Earth centre) [m].
    mu : float
        Gravitational parameter [m^3/s^2].

    Returns
    -------
    ndarray, shape (3,)
        Gravity gradient torque [N*m].
    """
    r_hat_x, r_hat_y, r_hat_z = r_body_unit
    coeff = 3.0 * mu / R**3

    tau_x = coeff * (Izz - Iyy) * r_hat_y * r_hat_z
    tau_y = coeff * (Ixx - Izz) * r_hat_z * r_hat_x
    tau_z = coeff * (Iyy - Ixx) * r_hat_x * r_hat_y

    return np.array([tau_x, tau_y, tau_z])


# ---------------------------------------------------------------------------
# Full Euler equations of motion
# ---------------------------------------------------------------------------

def euler_equations(t, state, Ixx, Iyy, Izz, torque_func):
    """Right-hand side for full rigid-body attitude dynamics.

    State = [omega_x, omega_y, omega_z, theta, phi, psi].

    The Euler rotational equations:
        Ixx * omega_x_dot = (Iyy - Izz) * omega_y * omega_z + tau_x
        Iyy * omega_y_dot = (Izz - Ixx) * omega_z * omega_x + tau_y
        Izz * omega_z_dot = (Ixx - Iyy) * omega_x * omega_y + tau_z

    Parameters
    ----------
    t : float
        Time [s].
    state : ndarray, shape (6,)
        [omega_x, omega_y, omega_z, theta, phi, psi] [rad/s, rad].
    Ixx, Iyy, Izz : float
        Principal moments of inertia [kg*m^2].
    torque_func : callable(t, state) -> ndarray(3,)
        External torque function returning [tau_x, tau_y, tau_z] [N*m].

    Returns
    -------
    ndarray, shape (6,)
        State derivative.
    """
    omega_x, omega_y, omega_z, theta, phi, psi = state
    omega = np.array([omega_x, omega_y, omega_z])

    # External torques
    tau = torque_func(t, state)

    # Euler's rotational equations
    omega_x_dot = ((Iyy - Izz) * omega_y * omega_z + tau[0]) / Ixx
    omega_y_dot = ((Izz - Ixx) * omega_z * omega_x + tau[1]) / Iyy
    omega_z_dot = ((Ixx - Iyy) * omega_x * omega_y + tau[2]) / Izz

    # Kinematic equations
    theta_dot, phi_dot, psi_dot = kinematic_equations(omega, theta, phi)

    return np.array([omega_x_dot, omega_y_dot, omega_z_dot,
                     theta_dot, phi_dot, psi_dot])


# ---------------------------------------------------------------------------
# Pitch-only dynamics (for chaos study)
# ---------------------------------------------------------------------------

def pitch_eom(t, state, n, Ixx, Iyy, Izz, tau_aero_func=None):
    """Simplified pitch-only equation of motion for chaos studies.

    theta_ddot = (Ixx - Izz)/Iyy * 3*n^2 * sin(theta)*cos(theta) + tau_aero/Iyy

    State = [theta, theta_dot].

    Parameters
    ----------
    t : float
        Time [s].
    state : ndarray, shape (2,)
        [theta, theta_dot] [rad, rad/s].
    n : float
        Mean motion [rad/s].
    Ixx, Iyy, Izz : float
        Principal moments of inertia [kg*m^2].
    tau_aero_func : callable(t, theta) -> float or None
        Aerodynamic torque about pitch axis [N*m].

    Returns
    -------
    ndarray, shape (2,)
        [theta_dot, theta_ddot].
    """
    theta, theta_dot = state

    # Gravity gradient contribution to pitch
    gg = (Ixx - Izz) / Iyy * 3.0 * n**2 * np.sin(theta) * np.cos(theta)

    # Aerodynamic torque
    if tau_aero_func is not None:
        tau_aero = tau_aero_func(t, theta)
    else:
        tau_aero = 0.0

    theta_ddot = gg + tau_aero / Iyy

    return np.array([theta_dot, theta_ddot])


# ---------------------------------------------------------------------------
# Attitude propagator class
# ---------------------------------------------------------------------------

class AttitudePropagator:
    """Propagates rigid-body attitude dynamics.

    Parameters
    ----------
    Ixx, Iyy, Izz : float
        Principal moments of inertia [kg*m^2].
    torque_func : callable(t, state) -> ndarray(3,) or None
        External torque function.  If None, only Euler coupling is active.
    """

    def __init__(self, Ixx=IXX, Iyy=IYY, Izz=IZZ, torque_func=None):
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        if torque_func is not None:
            self.torque_func = torque_func
        else:
            self.torque_func = lambda t, state: np.zeros(3)

    def propagate(self, state0, t_span, t_eval=None, method='RK45'):
        """Propagate full 3-axis attitude dynamics.

        Parameters
        ----------
        state0 : ndarray, shape (6,)
            [omega_x, omega_y, omega_z, theta, phi, psi] [rad/s, rad].
        t_span : tuple (t0, tf)
            Integration interval [s].
        t_eval : ndarray or None
            Output times.
        method : str
            Integration method for solve_ivp.

        Returns
        -------
        sol : OdeResult
            scipy solve_ivp solution.
        """
        def rhs(t, state):
            return euler_equations(t, state, self.Ixx, self.Iyy, self.Izz,
                                   self.torque_func)

        sol = solve_ivp(rhs, t_span, state0, method=method,
                        rtol=1e-10, atol=1e-12, max_step=DT_INTEGRATOR,
                        t_eval=t_eval, dense_output=True)
        return sol

    def propagate_pitch_only(self, state0, t_span, n, tau_aero_func=None,
                             t_eval=None, method='RK45'):
        """Propagate pitch-only dynamics for chaos studies.

        Parameters
        ----------
        state0 : ndarray, shape (2,)
            [theta, theta_dot] [rad, rad/s].
        t_span : tuple (t0, tf)
            Integration interval [s].
        n : float
            Mean motion [rad/s].
        tau_aero_func : callable(t, theta) -> float or None
            Aerodynamic pitch torque.
        t_eval : ndarray or None
            Output times.
        method : str
            Integration method.

        Returns
        -------
        sol : OdeResult
            scipy solve_ivp solution.
        """
        def rhs(t, state):
            return pitch_eom(t, state, n, self.Ixx, self.Iyy, self.Izz,
                             tau_aero_func)

        sol = solve_ivp(rhs, t_span, state0, method=method,
                        rtol=1e-10, atol=1e-12, max_step=DT_INTEGRATOR,
                        t_eval=t_eval, dense_output=True)
        return sol
