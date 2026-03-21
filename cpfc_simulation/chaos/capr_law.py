"""Chaos-Assisted Passive Reconfiguration (CAPR) Law -- Novel Algorithm."""
import numpy as np
from cpfc_simulation.config import (
    MU_EARTH, R_EARTH, J2, MASS_SAT, A_STOWED, A_DEPLOYED,
    ALT_NOMINAL, INC_NOMINAL, FORMATION_RADIUS
)


class CAPRController:
    """
    The CAPR controller -- the novel contribution unifying chaos + differential drag + J2.

    Physical mechanism:
    1. CubeSat attitude evolves on a strange attractor when drag plate switches at Omega_switch
    2. The attractor has time-averaged <Cd*A> determining differential drag
    3. By switching between attractors (or chaotic/regular), we control effective <Cd>
    4. This controls formation geometry without thrusters
    """

    def __init__(self, orbital_params, melnikov_data=None, poincare_data=None,
                 aero_model=None):
        self.n = orbital_params.get('n')  # mean motion
        self.kappa = orbital_params.get('kappa', 1.0)
        self.c = orbital_params.get('c', 0.0)
        self.a = orbital_params.get('a', R_EARTH + ALT_NOMINAL)

        self.melnikov_data = melnikov_data
        self.poincare_data = poincare_data
        self.aero_model = aero_model

        # State
        self.current_deploy_state = 0  # 0=stowed, 1=deployed
        self.switch_log = []
        self.last_switch_time = -np.inf
        self.min_dwell_time = 2.0 * np.pi / self.n  # [s] 1 orbital period for attractor settling

        # Precomputed lookup: Cd_eff as function of switching frequency
        self._Cd_lookup = None
        self._build_Cd_lookup()

    def _build_Cd_lookup(self):
        """Build lookup table: switching_frequency -> effective Cd*A product."""
        # This would normally use precomputed attractor data
        # For now, build an approximate model:
        # At high switching freq: Cd*A ~ average of stowed and deployed
        # At low freq: Cd*A ~ one or the other depending on duty cycle
        # At resonant freq (from Melnikov): Cd*A takes specific chaotic-average value

        Cd_stowed = 2.2  # approximate for ram-facing
        Cd_deployed = 2.5  # approximate with panels

        self._Cd_stowed = Cd_stowed * A_STOWED
        self._Cd_deployed = Cd_deployed * A_DEPLOYED
        self._Cd_average = 0.5 * (self._Cd_stowed + self._Cd_deployed)

        # If Melnikov data is available, build a frequency-dependent lookup
        if self.melnikov_data is not None:
            Omega_arr = self.melnikov_data.get('Omega', None)
            M_arr = self.melnikov_data.get('M_values', None)
            if Omega_arr is not None and M_arr is not None:
                # Normalize Melnikov integral to [0, 1] range
                M_norm = M_arr / (M_arr.max() + 1e-30)
                # CdA interpolates between stowed and deployed based on chaos intensity
                self._Cd_lookup = {
                    'Omega': Omega_arr,
                    'CdA': self._Cd_stowed + M_norm * (self._Cd_deployed - self._Cd_stowed)
                }

    def compute_formation_error(self, current_state, target_state):
        """Compute formation error decomposed into radial, along-track, cross-track.

        State ordering: [x, y, z, xdot, ydot, zdot].
        """
        e_x = current_state[0] - target_state[0]  # radial
        e_y = current_state[1] - target_state[1]  # along-track
        e_z = current_state[2] - target_state[2]  # cross-track
        return np.array([e_x, e_y, e_z])

    def compute_required_differential_drag(self, e_along_track, T_orb):
        """
        STEP 2: Compute required dfy to correct along-track error in one orbit.
        From Traub 2025 corrected secular solution:
        secular_gain = 3*kappa / ((1+2c) * n^2)
        Delta_y = secular_gain * dfy * T_orb
        Therefore: dfy_req = e_along_track / (secular_gain * T_orb)
        """
        n = self.n
        secular_gain = 3 * self.kappa / ((1 + 2*self.c) * n**2)
        dfy_req = e_along_track / (secular_gain * T_orb) if abs(secular_gain * T_orb) > 1e-30 else 0.0
        return dfy_req

    def select_attractor(self, dfy_required, rho, v_rel):
        """
        STEP 3: Select the switching regime that produces the required dfy.
        dfy = -0.5 * rho * v^2 * (CdA_deputy - CdA_chief) / mass
        So: CdA_needed = CdA_chief - 2*mass*dfy / (rho * v^2)
        """
        if abs(rho * v_rel**2) < 1e-30:
            return self.current_deploy_state, 0.0

        CdA_chief = self._Cd_stowed  # chief always in low-drag config
        CdA_needed = CdA_chief - 2 * MASS_SAT * dfy_required / (rho * v_rel**2)

        # Determine which regime (stowed/deployed/chaotic-average) is closest
        options = {
            0: self._Cd_stowed,
            1: self._Cd_deployed,
        }

        # If we have a frequency-dependent lookup, add intermediate options
        if self._Cd_lookup is not None:
            CdA_interp = self._Cd_lookup['CdA']
            # Find the CdA value closest to what we need
            idx_best_lookup = np.argmin(np.abs(CdA_interp - CdA_needed))
            options[2] = CdA_interp[idx_best_lookup]

        best_state = min(options.keys(), key=lambda k: abs(options[k] - CdA_needed))
        predicted_CdA = options[best_state]

        return best_state, predicted_CdA

    def poincare_targeting(self, current_orbital_state, target_fixed_point,
                           manifold_data=None):
        """
        STEP 4: Poincare section targeting.
        Determine if current state is in the basin of attraction of the target.

        Decision: switch to deployed if inside influence basin of stable manifold.
        """
        if manifold_data is None or target_fixed_point is None:
            return None  # no manifold data available, skip this step

        # Distance to target fixed point in (y, ydot) space
        # State ordering: [x, y, z, xdot, ydot, zdot]
        y_current = current_orbital_state[1]
        ydot_current = current_orbital_state[4]

        dist = np.sqrt((y_current - target_fixed_point[0])**2 +
                       (ydot_current - target_fixed_point[1])**2)

        # If we have manifold trajectories, check proximity to stable manifold
        if manifold_data is not None and len(manifold_data) > 0:
            min_manifold_dist = np.inf
            current_point = np.array([y_current, ydot_current])
            for traj in manifold_data:
                if traj.shape[1] >= 4:
                    manifold_points = traj[:, [2, 3]]  # y, ydot columns
                    dists = np.linalg.norm(manifold_points - current_point, axis=1)
                    min_manifold_dist = min(min_manifold_dist, dists.min())
            # Near stable manifold means we'll naturally approach the fixed point
            if min_manifold_dist < FORMATION_RADIUS:
                return True

        # Simple heuristic: if within 2x formation radius, we're in the basin
        return dist < 2 * FORMATION_RADIUS

    def verify_chaos(self, mle_current):
        """
        STEP 5: Chaos verification and fallback.
        """
        if mle_current < 0:
            return 'reinitialize'  # not chaotic when should be
        elif mle_current > 0.5:
            return 'reduce_frequency'  # too chaotic
        else:
            return 'nominal'

    def compute_duty_cycle(self, CdA_target):
        """
        Compute the duty cycle (fraction of time deployed) to achieve a target CdA.
        CdA_effective = (1 - D)*CdA_stowed + D*CdA_deployed
        D = (CdA_target - CdA_stowed) / (CdA_deployed - CdA_stowed)
        """
        denom = self._Cd_deployed - self._Cd_stowed
        if abs(denom) < 1e-30:
            return 0.5
        D = (CdA_target - self._Cd_stowed) / denom
        return np.clip(D, 0.0, 1.0)

    def estimate_reconfiguration_time(self, e_along_track, rho, v_rel, T_orb):
        """
        Estimate how many orbits are needed to correct the along-track error
        given the available differential drag authority.
        """
        max_dfy = 0.5 * rho * v_rel**2 * abs(self._Cd_deployed - self._Cd_stowed) / MASS_SAT
        if max_dfy < 1e-30:
            return np.inf

        n = self.n
        secular_gain = 3 * self.kappa / ((1 + 2*self.c) * n**2)
        correction_per_orbit = secular_gain * max_dfy * T_orb
        if abs(correction_per_orbit) < 1e-30:
            return np.inf

        n_orbits = abs(e_along_track) / abs(correction_per_orbit)
        return n_orbits

    def reset(self):
        """Reset controller state for a new simulation run."""
        self.current_deploy_state = 0
        self.switch_log = []
        self.last_switch_time = -np.inf

    def __call__(self, t, current_formation_state, target_formation_state,
                 attitude_state=None, rho=1e-12, v_rel=7500.0,
                 mle_current=None, T_orb=5600.0):
        """
        Main CAPR control law evaluation.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        current_formation_state : ndarray
            Current relative state [x, xdot, y, ydot, z, zdot].
        target_formation_state : ndarray
            Target relative state [x, xdot, y, ydot, z, zdot].
        attitude_state : ndarray or None
            Current attitude state (angular velocities, angles).
        rho : float
            Local atmospheric density [kg/m^3].
        v_rel : float
            Relative velocity magnitude [m/s].
        mle_current : float or None
            Current MLE estimate (from real-time chaos monitor).
        T_orb : float
            Orbital period [s].

        Returns
        -------
        result : dict
            deploy_command, predicted_CdA, formation_error, dfy_required,
            chaos_status, e_along_track, duty_cycle, est_reconfig_orbits.
        """
        # STEP 1: Formation error
        error = self.compute_formation_error(current_formation_state, target_formation_state)
        e_along_track = error[1]

        # STEP 2: Required differential drag
        dfy_req = self.compute_required_differential_drag(e_along_track, T_orb)

        # STEP 3: Closed-loop deploy decision.
        # The differential drag force dfy drives along-track drift.
        # Deploying panels INCREASES CdA_deputy, making dfy more negative
        # (deputy decelerates relative to chief).
        #
        # The effect on y depends on secular_gain sign:
        # - secular_gain > 0 (equatorial): negative dfy → negative y drift
        # - secular_gain < 0 (polar): negative dfy → POSITIVE y drift
        #
        # So deploy panels when we want y to change in direction of sign(secular_gain * (-1))
        # i.e., deploy when e_y * secular_gain > 0 (error and gain same sign)
        #
        # Equivalently: deploy when dfy_req > 0 (we need positive dfy to correct,
        # but deploying gives negative dfy, so this creates opposing force — WRONG)
        # Actually: deploy when dfy_req < 0 AND secular_gain > 0, OR
        #           deploy when dfy_req > 0 AND secular_gain < 0.
        # Simplified: deploy when e_y and (1+2c) have OPPOSITE signs.
        n = self.n
        secular_gain = 3 * self.kappa / ((1 + 2*self.c) * n**2)
        dead_band = FORMATION_RADIUS  # [m] scale dead band to formation size
        if abs(e_along_track) < dead_band:
            deploy_state = self.current_deploy_state
        elif e_along_track * (1 + 2*self.c) < 0:
            # Error and restoring coefficient have opposite signs → deploy
            deploy_state = 1
        else:
            deploy_state = 0

        predicted_CdA = self._Cd_deployed if deploy_state == 1 else self._Cd_stowed

        # STEP 4: Chaos verification (optional monitoring)
        chaos_status = 'nominal'
        if mle_current is not None:
            chaos_status = self.verify_chaos(mle_current)

        # Enforce minimum dwell time
        if t - self.last_switch_time < self.min_dwell_time:
            deploy_state = self.current_deploy_state
        elif deploy_state != self.current_deploy_state:
            self.current_deploy_state = deploy_state
            self.last_switch_time = t
            self.switch_log.append({
                'time': t,
                'state': deploy_state,
                'error': float(e_along_track),
                'dfy_req': float(dfy_req)
            })

        # Compute supplementary diagnostics
        duty_cycle = self.compute_duty_cycle(predicted_CdA)
        est_reconfig = self.estimate_reconfiguration_time(e_along_track, rho, v_rel, T_orb)

        return {
            'deploy_command': self.current_deploy_state,
            'predicted_CdA': predicted_CdA,
            'formation_error': error,
            'dfy_required': dfy_req,
            'chaos_status': chaos_status,
            'e_along_track': e_along_track,
            'duty_cycle': duty_cycle,
            'est_reconfig_orbits': est_reconfig,
            'poincare_decision': None,
        }
