"""Coupled attitude-orbital dynamics for CPFC.

Implements the full closed-loop coupling where:
  attitude (theta) -> angle of attack (alpha) -> Cd(alpha) via Sentman ->
  drag force -> differential drag -> orbital relative motion ->
  aerodynamic torque -> attitude dynamics

This is the core physics that makes the CAPR algorithm work.
"""
import numpy as np
from scipy.integrate import solve_ivp
from cpfc_simulation.config import (
    MU_EARTH, R_EARTH, J2, IXX, IYY, IZZ, MASS_SAT,
    L_Z, PANEL_LENGTH, PANEL_WIDTH, A_STOWED, A_DEPLOYED,
    OMEGA_EARTH, DT_INTEGRATOR
)
from cpfc_simulation.dynamics.perturbations import Cd_freemolecular
from cpfc_simulation.dynamics.aerodynamic_model import CubeSatAeroModel
from cpfc_simulation.dynamics.relative_motion import compute_SS_coefficients


class CoupledDynamics:
    """Full coupled attitude-orbital dynamics for one deputy satellite.

    The coupling mechanism:
    1. Current pitch angle theta determines angle of attack alpha
       (alpha = theta for small perturbations from ram-pointing)
    2. alpha determines Cd via Sentman free-molecular model
    3. Cd * A determines drag force, and differential drag vs chief
    4. Differential drag drives along-track relative motion (dfy)
    5. Gravity gradient + aerodynamic torque drive attitude (back to step 1)
    """

    def __init__(self, a, inc_rad, rho, v_orb, panel_deployed=False,
                 Cd_chief=None, A_chief=A_STOWED, speed_ratio=5.0,
                 T_exo=1000.0):
        """
        Parameters
        ----------
        a : float
            Semi-major axis [m].
        inc_rad : float
            Inclination [rad].
        rho : float
            Atmospheric density [kg/m^3].
        v_orb : float
            Orbital velocity [m/s].
        panel_deployed : bool
            Whether deputy drag plates are deployed.
        Cd_chief : float or None
            Chief drag coefficient. If None, computed for ram-pointing (alpha=0).
        A_chief : float
            Chief cross-sectional area [m^2].
        speed_ratio : float
            Molecular speed ratio v/v_thermal.
        T_exo : float
            Exospheric temperature [K] for Sentman model.
        """
        self.n, self.kappa, self.c, self.s = compute_SS_coefficients(a, inc_rad)
        self.a = a
        self.rho = rho
        self.v_orb = v_orb
        self.panel_deployed = panel_deployed
        self.speed_ratio = speed_ratio
        self.T_exo = T_exo
        self.aero_model = CubeSatAeroModel()

        # Chief CdA: if provided use it, otherwise compute for ram-pointing.
        # In practice, set CdA_chief to the attractor-averaged CdA so that
        # differential drag = 0 when both satellites have the same attitude
        # dynamics. Only panel switching creates a difference.
        if Cd_chief is not None:
            self.CdA_chief = Cd_chief  # Cd_chief is actually CdA_chief when provided
        else:
            self.CdA_chief = Cd_freemolecular(1e-6, speed_ratio, 300.0, T_exo) * A_chief
        self.A_chief = A_chief

        # Atmospheric co-rotation correction (from astro21-sim):
        # v_rel = v_orbital - omega_earth × r ≈ v_orb - omega*r*cos(i)
        # At 320 km, omega*r ~ 487 m/s, so this reduces drag by ~12%.
        self.v_rel = np.sqrt(v_orb**2 + (OMEGA_EARTH * a)**2
                             - 2.0 * v_orb * OMEGA_EARTH * a * np.cos(inc_rad))

        # Gravity gradient parameter
        self.gg_coeff = 3.0 * self.n**2 * (IXX - IZZ) / IYY

        # Panel offset for aerodynamic torque
        self.L_offset = L_Z / 2 + PANEL_LENGTH / 2

    def compute_Cd_from_attitude(self, theta):
        """Compute instantaneous Cd*A from current pitch angle.

        theta IS the angle of attack for a satellite in LVLH frame
        (alpha = theta when the satellite's body frame Z-axis was
        initially aligned with the velocity vector).
        """
        alpha = theta  # angle of attack = pitch angle from ram
        Cd, A_eff, CdA = self.aero_model.effective_drag(
            alpha, self.speed_ratio, self.panel_deployed
        )
        return Cd, A_eff, CdA

    def compute_chief_CdA(self, theta_chief):
        """Compute chief's instantaneous CdA from its pitch angle (always stowed)."""
        alpha = theta_chief
        Cd, A_eff, CdA = self.aero_model.effective_drag(
            alpha, self.speed_ratio, panel_deployed=False  # chief always stowed
        )
        return CdA

    def compute_differential_drag(self, theta_dep, theta_chief=None):
        """Compute differential specific force dfy from attitude-driven Cd.

        dfy = -0.5 * rho * v^2 * (CdA_deputy - CdA_chief) / mass

        This is THE coupling: attitude -> drag -> orbit.

        When theta_chief is provided, the chief's CdA is computed from
        its own tumbling angle (fair comparison). Otherwise falls back
        to the constant CdA_chief.
        """
        _, _, CdA_deputy = self.compute_Cd_from_attitude(theta_dep)
        if theta_chief is not None:
            CdA_chief = self.compute_chief_CdA(theta_chief)
        else:
            CdA_chief = self.CdA_chief
        dfy = -0.5 * self.rho * self.v_rel**2 * (CdA_deputy - CdA_chief) / MASS_SAT
        return dfy, CdA_deputy

    def compute_aero_torque(self, theta):
        """Compute aerodynamic pitch torque from current attitude.

        This closes the loop: orbit conditions -> torque -> attitude.
        """
        return self.aero_model.aero_torque(
            theta, self.rho, self.v_rel, self.speed_ratio, self.panel_deployed
        )

    def compute_chief_aero_torque(self, theta_chief):
        """Compute chief's aerodynamic pitch torque (always stowed)."""
        return self.aero_model.aero_torque(
            theta_chief, self.rho, self.v_rel, self.speed_ratio,
            panel_deployed=False  # chief always stowed
        )

    def eom(self, t, state):
        """Coupled equations of motion.

        State = [theta_dep, theta_dot_dep,
                 theta_chief, theta_dot_chief,
                 x, y, z, xdot, ydot, zdot]

        Deputy attitude:
            theta_dep_ddot = gg_torque(theta_dep) + aero_torque(theta_dep, panel_state)

        Chief attitude (always stowed, same orbit environment):
            theta_chief_ddot = gg_torque(theta_chief) + aero_torque(theta_chief, stowed)

        Orbital (corrected SS):
            xddot = 2*n*kappa*ydot + (1+2c)*n^2*x + dfx
            yddot = -2*n*kappa*xdot + dfy(theta_dep, theta_chief)  <-- COUPLED
            zddot = -n^2*s*z
        """
        theta_dep, theta_dot_dep = state[0], state[1]
        theta_chief, theta_dot_chief = state[2], state[3]
        x, y, z = state[4], state[5], state[6]
        xdot, ydot, zdot = state[7], state[8], state[9]

        n, kappa, c_ss, s_ss = self.n, self.kappa, self.c, self.s

        # --- Deputy attitude dynamics ---
        tau_gg_dep = self.gg_coeff * np.sin(theta_dep) * np.cos(theta_dep)
        tau_aero_dep = self.compute_aero_torque(theta_dep) / IYY
        theta_dep_ddot = tau_gg_dep + tau_aero_dep

        # --- Chief attitude dynamics (same torques, but always stowed) ---
        tau_gg_chief = self.gg_coeff * np.sin(theta_chief) * np.cos(theta_chief)
        tau_aero_chief = self.compute_chief_aero_torque(theta_chief) / IYY
        theta_chief_ddot = tau_gg_chief + tau_aero_chief

        # --- Orbital dynamics (SS with attitude-coupled drag) ---
        # Differential drag from BOTH instantaneous attitudes
        dfy, _ = self.compute_differential_drag(theta_dep, theta_chief)

        xddot = 2.0 * n * kappa * ydot + (1.0 + 2.0 * c_ss) * n**2 * x
        yddot = -2.0 * n * kappa * xdot + dfy
        zddot = -n**2 * s_ss * z

        return np.array([theta_dot_dep, theta_dep_ddot,
                         theta_dot_chief, theta_chief_ddot,
                         xdot, ydot, zdot,
                         xddot, yddot, zddot])

    def propagate(self, state0, t_span, t_eval=None, max_step=1.0):
        """Propagate coupled system.

        Parameters
        ----------
        state0 : ndarray, shape (10,)
            [theta_dep, theta_dot_dep, theta_chief, theta_dot_chief,
             x, y, z, xdot, ydot, zdot]
        t_span : tuple
            (t0, tf)
        t_eval : ndarray or None
            Output times.
        max_step : float
            Maximum integration step [s].

        Returns
        -------
        sol : OdeResult
        """
        sol = solve_ivp(self.eom, t_span, state0, method='RK45',
                        rtol=1e-10, atol=1e-12, max_step=max_step,
                        t_eval=t_eval, dense_output=True)
        return sol

    def set_panel_state(self, deployed):
        """Change panel deployment state (called by controller)."""
        self.panel_deployed = deployed

    def set_density(self, rho):
        """Update atmospheric density (e.g., from NRLMSISE-00 refresh)."""
        self.rho = rho


def run_coupled_simulation(orbital_params, rho, T_sim, dt_output,
                            panel_schedule=None, initial_attitudes=None,
                            n_deputies=3, formation_radius=500.0,
                            rng_seed=12345):
    """Run full coupled attitude-orbital simulation for all deputies.

    Parameters
    ----------
    orbital_params : dict
        From compute_orbital_params(). Keys: n, kappa, c, s, a, inc.
    rho : float
        Atmospheric density [kg/m^3].
    T_sim : float
        Simulation duration [s].
    dt_output : float
        Output time step [s].
    panel_schedule : callable(t, deputy_idx, state) -> bool, or None
        Returns True if panels should be deployed at time t for deputy deputy_idx.
        If None, panels stay stowed.
    initial_attitudes : ndarray (n_deputies, 2) or None
        Initial [theta, theta_dot] for each deputy.
    n_deputies : int
        Number of deputy satellites.
    formation_radius : float
        PCO formation radius [m].
    rng_seed : int
        Random seed for initial perturbations.

    Returns
    -------
    dict with:
        time: (N_t,) array
        states: (N_t, n_deputies, 8) array [theta, thetadot, x, y, z, xdot, ydot, zdot]
        CdA_history: (N_t, n_deputies) array
        dfy_history: (N_t, n_deputies) array
    """
    from cpfc_simulation.formation.formation_geometry import pco_formation_state

    n = orbital_params['n']
    kappa = orbital_params['kappa']
    a = orbital_params['a']
    inc_rad = orbital_params['inc']
    v_orb = n * a

    t_output = np.arange(0, T_sim, dt_output)
    N_t = len(t_output)

    # Molecular speed ratio: v_orb / v_thermal
    # v_thermal = sqrt(2*k*T/m) ~ sqrt(2*1.38e-23*1000/28e-3*6e23) ~ 500 m/s for N2 at 1000K
    # Actually: v_thermal = sqrt(2*k_B*T/m_molecule)
    # For atomic oxygen (dominant at 450km): m = 16 amu = 16*1.66e-27 kg
    # v_thermal = sqrt(2*1.38e-23*1000/(16*1.66e-27)) ~ 1300 m/s
    # speed_ratio = v_orb / v_thermal ~ 7500/1300 ~ 5.8
    speed_ratio = v_orb / 1300.0  # approximate

    # Target formation at t=0
    target_t0 = pco_formation_state(0.0, n, kappa, formation_radius, n_deputies + 1)
    target_dep = target_t0[1:]  # (n_deputies, 6)

    # Initial orbital states with perturbation
    rng = np.random.default_rng(rng_seed)
    init_orb = target_dep.copy()
    init_orb[:, [0, 2, 4]] += rng.normal(0, 20, (n_deputies, 3))
    init_orb[:, [1, 3, 5]] += rng.normal(0, 0.01, (n_deputies, 3))

    # Initial attitude states
    if initial_attitudes is None:
        initial_attitudes = np.zeros((n_deputies, 2))
        initial_attitudes[:, 0] = rng.uniform(-0.3, 0.3, n_deputies)  # theta
        initial_attitudes[:, 1] = rng.uniform(-0.01, 0.01, n_deputies)  # theta_dot

    # Create coupled dynamics objects per deputy
    systems = []
    for j in range(n_deputies):
        deployed = False
        if panel_schedule is not None:
            deployed = panel_schedule(0.0, j, None)
        sys = CoupledDynamics(a, inc_rad, rho, v_orb,
                               panel_deployed=deployed,
                               speed_ratio=speed_ratio)
        systems.append(sys)

    # Storage: state = [theta, theta_dot, x, y, z, xdot, ydot, zdot]
    states_history = np.zeros((N_t, n_deputies, 8))
    CdA_history = np.zeros((N_t, n_deputies))
    dfy_history = np.zeros((N_t, n_deputies))

    # Set initial states
    for j in range(n_deputies):
        states_history[0, j, 0] = initial_attitudes[j, 0]  # theta
        states_history[0, j, 1] = initial_attitudes[j, 1]  # theta_dot
        states_history[0, j, 2:8] = init_orb[j]  # x, y, z, xdot, ydot, zdot

    # Propagate
    for k in range(1, N_t):
        t_now = t_output[k - 1]
        dt = t_output[k] - t_output[k - 1]

        for j in range(n_deputies):
            state_j = states_history[k-1, j].copy()

            # Update panel state from schedule
            if panel_schedule is not None:
                deployed = panel_schedule(t_now, j, state_j)
                systems[j].set_panel_state(deployed)

            # Propagate coupled system
            sol = systems[j].propagate(state_j, (0, dt), max_step=min(dt, 2.0))

            if sol.success and sol.y.shape[1] > 0:
                states_history[k, j] = sol.y[:, -1]
            else:
                states_history[k, j] = state_j  # fallback

            # Record CdA and dfy
            theta_k = states_history[k, j, 0]
            _, _, CdA = systems[j].compute_Cd_from_attitude(theta_k)
            dfy, _ = systems[j].compute_differential_drag(theta_k)
            CdA_history[k, j] = CdA
            dfy_history[k, j] = dfy

    return {
        'time': t_output,
        'states': states_history,
        'CdA_history': CdA_history,
        'dfy_history': dfy_history,
    }
