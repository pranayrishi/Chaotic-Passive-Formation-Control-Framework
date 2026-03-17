"""Benchmark controllers for comparison against CAPR."""
import numpy as np
from scipy.optimize import linprog, minimize


class LPDifferentialDragController:
    """
    Controller A: Linear Programming differential drag.
    From 'Small Satellite Constellation Separation using LP'.
    """
    def __init__(self, dfy_min, dfy_max, n, kappa, c):
        self.dfy_min = dfy_min
        self.dfy_max = dfy_max
        self.n = n
        self.kappa = kappa
        self.c = c
        self.switch_log = []

    def __call__(self, t, current_state, target_state, T_orb):
        """Compute optimal drag plate command via LP."""
        e_y = current_state[1] - target_state[1]
        secular_gain = 3*self.kappa / ((1+2*self.c) * self.n**2)
        dfy_req = e_y / (secular_gain * T_orb) if abs(secular_gain * T_orb) > 1e-30 else 0.0

        # Clamp to achievable range
        dfy_cmd = np.clip(dfy_req, self.dfy_min, self.dfy_max)

        # Binary: deploy if we need more drag on deputy than chief
        deploy = 1 if dfy_cmd < 0 else 0  # negative dfy = deputy needs more drag

        self.switch_log.append({'time': t, 'deploy': deploy, 'dfy_req': dfy_req})
        return deploy, dfy_cmd


class ConvexOptController:
    """
    Controller B: Convex optimization of relative orbit maneuvers.
    Solves a relaxed binary program then rounds.
    """
    def __init__(self, dfy_min, dfy_max, n, kappa, c, horizon_orbits=5):
        self.dfy_min = dfy_min
        self.dfy_max = dfy_max
        self.n = n
        self.kappa = kappa
        self.c = c
        self.horizon = horizon_orbits
        self.switch_log = []

    def __call__(self, t, current_state, target_state, T_orb):
        """Solve convex relaxation for optimal switching sequence."""
        e_y = current_state[1] - target_state[1]
        secular_gain = 3*self.kappa / ((1+2*self.c) * self.n**2)

        N = self.horizon
        # Decision variable: u_i in [0,1] for each orbit (relaxed binary)
        # Objective: min sum |u_i - 0.5| ≈ min switching
        # Constraint: sum(secular_gain * (dfy_max*u_i + dfy_min*(1-u_i)) * T_orb) corrects e_y

        dfy_range = self.dfy_max - self.dfy_min

        # LP: min c^T u, s.t. A_eq u = b_eq, 0 <= u <= 1
        # We approximate |u - 0.5| with linear relaxation
        c_obj = np.ones(N)  # minimize total deployment (proxy)

        # Constraint: total drift correction
        if abs(secular_gain * dfy_range * T_orb) > 1e-30:
            A_eq = np.ones((1, N)) * secular_gain * dfy_range * T_orb
            b_eq = np.array([e_y - N * secular_gain * self.dfy_min * T_orb])

            bounds = [(0, 1)] * N
            try:
                res = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
                if res.success:
                    u_opt = res.x
                    deploy = 1 if u_opt[0] > 0.5 else 0
                    dfy_cmd = self.dfy_min + u_opt[0] * dfy_range
                    self.switch_log.append({'time': t, 'deploy': deploy})
                    return deploy, dfy_cmd
            except Exception:
                pass

        # Fallback to simple threshold
        deploy = 1 if e_y > 0 else 0
        dfy_cmd = self.dfy_max if deploy else self.dfy_min
        self.switch_log.append({'time': t, 'deploy': deploy})
        return deploy, dfy_cmd


class ConstraintTighteningController:
    """
    Controller C: Constraint tightening for satellite formations.
    Adds safety margins that tighten over the prediction horizon.
    """
    def __init__(self, dfy_min, dfy_max, n, kappa, c, safety_margin=50.0):
        self.dfy_min = dfy_min
        self.dfy_max = dfy_max
        self.n = n
        self.kappa = kappa
        self.c = c
        self.safety_margin = safety_margin  # [m]
        self.switch_log = []

    def __call__(self, t, current_state, target_state, T_orb):
        """Apply constraint-tightened control."""
        e_y = current_state[1] - target_state[1]
        secular_gain = 3*self.kappa / ((1+2*self.c) * self.n**2)

        # Tighten constraint: require error < target - margin
        e_tightened = e_y - np.sign(e_y) * self.safety_margin

        dfy_req = e_tightened / (secular_gain * T_orb) if abs(secular_gain * T_orb) > 1e-30 else 0.0
        dfy_cmd = np.clip(dfy_req, self.dfy_min, self.dfy_max)

        deploy = 1 if dfy_cmd < -abs(self.dfy_min) * 0.3 else 0
        self.switch_log.append({'time': t, 'deploy': deploy, 'dfy_cmd': dfy_cmd})
        return deploy, dfy_cmd


class ActiveThrusterController:
    """
    Controller D: Ideal active thruster (oracle benchmark).
    Applies exact required force — no mechanical constraints.
    """
    def __init__(self, n, kappa, c, mass=4.0):
        self.n = n
        self.kappa = kappa
        self.c = c
        self.mass = mass
        self.total_dv = 0.0
        self.thrust_log = []

    def __call__(self, t, current_state, target_state, T_orb, dt=1.0):
        """Apply exact correction thrust."""
        e_y = current_state[1] - target_state[1]
        secular_gain = 3*self.kappa / ((1+2*self.c) * self.n**2)

        dfy_req = e_y / (secular_gain * T_orb) if abs(secular_gain * T_orb) > 1e-30 else 0.0

        # Accumulate delta-V
        dv = abs(dfy_req) * dt
        self.total_dv += dv
        self.thrust_log.append({'time': t, 'dfy': dfy_req, 'dv': dv})

        return dfy_req
