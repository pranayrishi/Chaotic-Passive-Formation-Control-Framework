"""Free molecular flow aerodynamic model for 3U CubeSat with deployable panels.

Based on Sentman (1961) and Aslanov & Sizov (2021) for torque modelling.
"""
import numpy as np
from cpfc_simulation.config import (
    MASS_SAT, L_X, L_Y, L_Z, A_STOWED, A_DEPLOYED,
    PANEL_LENGTH, PANEL_WIDTH
)
from cpfc_simulation.dynamics.perturbations import Cd_freemolecular


def _Cn_freemolecular(alpha, s, Tw=300.0, Ti=1000.0):
    """Sentman normal force coefficient for a flat plate in free molecular flow.

    Parameters
    ----------
    alpha : float
        Angle of attack [rad].
    s : float
        Molecular speed ratio.
    Tw : float
        Wall temperature [K].
    Ti : float
        Incident gas temperature [K].

    Returns
    -------
    float
        Normal force coefficient Cn.
    """
    from scipy.special import erf

    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)
    gamma = s * cos_a
    sqrt_pi = np.sqrt(np.pi)
    sigma_n = 1.0  # fully diffuse

    # Pressure coefficient (normal momentum transfer)
    Cp = (1.0 / (s**2 * sqrt_pi)) * (
        gamma * sqrt_pi * (1.0 + erf(gamma))
        * (0.5 + gamma**2 + 0.5 * sigma_n * np.sqrt(Tw / Ti))
        + np.exp(-gamma**2) * (1.0 + 0.5 * sigma_n * np.sqrt(Tw / Ti))
    )

    # Shear coefficient (tangential momentum transfer)
    P = np.exp(-gamma**2) + sqrt_pi * gamma * (1.0 + erf(gamma))
    Ctau = (sin_a / (s * sqrt_pi)) * P * sigma_n

    # Normal force coefficient: component perpendicular to surface
    # Cn = Cp (pressure is along normal) - Ctau projected onto normal = 0
    # Actually Cn is the force coefficient normal to the flow direction:
    # Cn = Cp * sin(alpha) - Ctau * cos(alpha)
    Cn = Cp * sin_a - Ctau * cos_a

    return Cn


class CubeSatAeroModel:
    """Computes effective Cd and cross-sectional area as function of attitude
    and panel state for a 3U CubeSat with two deployable drag panels.
    """

    def __init__(self, mass=MASS_SAT):
        self.mass = mass
        # Panel offset from geometric center (panel COP is at panel midpoint)
        self.panel_offset = L_Z / 2.0 + PANEL_LENGTH / 2.0  # [m]

    def effective_drag(self, alpha, speed_ratio, panel_deployed=False):
        """Compute effective Cd*A product for current attitude and panel state.

        Parameters
        ----------
        alpha : float
            Angle of attack [rad].
        speed_ratio : float
            Molecular speed ratio v_bulk / v_thermal.
        panel_deployed : bool
            Whether drag panels are deployed.

        Returns
        -------
        Cd_eff : float
            Effective drag coefficient.
        A_eff : float
            Effective cross-sectional area [m^2].
        CdA : float
            Product Cd * A [m^2].
        """
        # Body projected area as function of angle of attack
        # Body is a rectangular prism L_X x L_Y x L_Z
        # Flow nominally along -Z (velocity direction), alpha rotates about Y
        A_body = abs(L_X * L_Z * np.sin(alpha)) + abs(L_X * L_Y * np.cos(alpha))
        Cd_body = Cd_freemolecular(max(abs(alpha), 1e-6), speed_ratio)

        if panel_deployed:
            # Two panels, each PANEL_LENGTH x PANEL_WIDTH, perpendicular to flow
            # when alpha=0. At angle alpha, projected area changes.
            A_panel = 2.0 * PANEL_LENGTH * PANEL_WIDTH * abs(np.cos(alpha))
            Cd_panel = Cd_freemolecular(max(abs(alpha), 1e-6), speed_ratio)
            A_eff = A_body + A_panel
            if A_eff > 1e-12:
                Cd_eff = (Cd_body * A_body + Cd_panel * A_panel) / A_eff
            else:
                Cd_eff = Cd_body
        else:
            A_eff = A_body
            Cd_eff = Cd_body

        return Cd_eff, A_eff, Cd_eff * A_eff

    def aero_torque(self, alpha, rho, v_rel_mag, speed_ratio, panel_deployed=False):
        """Aerodynamic restoring torque about pitch axis.

        Computed from the offset between the centre of pressure and the
        centre of mass.  The dominant contribution comes from the deployed
        panels whose centre of pressure is displaced along the body Z-axis
        by ``panel_offset`` from the CG.

        Based on Aslanov & Sizov (2021).

        Parameters
        ----------
        alpha : float
            Angle of attack [rad].
        rho : float
            Atmospheric density [kg/m^3].
        v_rel_mag : float
            Relative velocity magnitude [m/s].
        speed_ratio : float
            Molecular speed ratio.
        panel_deployed : bool
            Whether drag panels are deployed.

        Returns
        -------
        float
            Aerodynamic torque about pitch axis [N*m].
            Negative torque is restoring (for positive alpha).
        """
        q = 0.5 * rho * v_rel_mag**2  # dynamic pressure

        # --- Body torque contribution ---
        # The body CG is at the geometric centre, and for a symmetric prism
        # the CP coincides with the CG to first order, so body torque ~ 0.
        # A small restoring contribution exists from the asymmetry in normal
        # force distribution along the body length.
        Cn_body = _Cn_freemolecular(max(abs(alpha), 1e-6), speed_ratio)
        # Body CP offset from CG (approximately L_Z/6 for linear pressure)
        body_cp_offset = L_Z / 6.0
        A_body_normal = L_X * L_Z  # area of the face generating normal force
        tau_body = -q * A_body_normal * Cn_body * body_cp_offset * np.sign(alpha)

        if not panel_deployed:
            return tau_body

        # --- Panel torque contribution ---
        # Each panel is at offset +/- panel_offset along body Z.
        # The normal force coefficient acts on each panel.
        Cn_panel = _Cn_freemolecular(max(abs(alpha), 1e-6), speed_ratio)
        A_single_panel = PANEL_LENGTH * PANEL_WIDTH

        # For a symmetric pair of panels (one fore, one aft of CG),
        # the net torque comes from the difference in projected areas
        # at angle of attack.  The fore panel sees a different effective
        # alpha than the aft panel due to the body wake, but in free
        # molecular flow there is no wake.  The torque arises because
        # the normal force on each panel creates a moment arm about CG.
        #
        # Panel at +L_offset: force generates negative torque (restoring)
        # Panel at -L_offset: force generates positive torque (restoring)
        # Net restoring torque = -2 * q * A_panel * Cn * L_offset * sin(alpha)
        # (factor of sin(alpha) because the moment arm projection changes)
        tau_panel = (-2.0 * q * A_single_panel * Cn_panel
                     * self.panel_offset * np.sin(alpha))

        return tau_body + tau_panel

    def ballistic_coefficient(self, alpha, speed_ratio, panel_deployed=False):
        """Compute ballistic coefficient B = m / (Cd * A).

        Parameters
        ----------
        alpha : float
            Angle of attack [rad].
        speed_ratio : float
            Molecular speed ratio.
        panel_deployed : bool
            Whether drag panels are deployed.

        Returns
        -------
        float
            Ballistic coefficient [kg/m^2].
        """
        _, _, CdA = self.effective_drag(alpha, speed_ratio, panel_deployed)
        if CdA < 1e-15:
            return np.inf
        return self.mass / CdA
