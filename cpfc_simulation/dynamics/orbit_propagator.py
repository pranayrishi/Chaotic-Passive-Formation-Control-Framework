"""High-fidelity absolute orbit propagator in ECI frame.

Includes J2+J3+J4 zonal harmonics, NRLMSISE-00 atmospheric drag,
and cannonball SRP with cylindrical shadow model.
"""
import math
import numpy as np
from datetime import timedelta
from scipy.integrate import solve_ivp
from cpfc_simulation.config import MU_EARTH, DT_INTEGRATOR, AU, EPOCH, CR
from cpfc_simulation.dynamics.perturbations import (
    accel_J2, accel_J3, accel_J4, accel_drag, accel_SRP, get_atmospheric_density
)


class OrbitPropagator:
    """Propagates absolute ECI orbit with full perturbation model.

    Parameters
    ----------
    Cd : float
        Drag coefficient.
    A_cross : float
        Cross-sectional area [m^2].
    mass : float
        Satellite mass [kg].
    epoch : datetime or None
        Mission epoch (UTC). Defaults to config.EPOCH.
    include_drag : bool
        Include atmospheric drag.
    include_SRP : bool
        Include solar radiation pressure.
    include_J3J4 : bool
        Include J3 and J4 harmonics.
    """

    def __init__(self, Cd=2.2, A_cross=0.01, mass=4.0, epoch=None,
                 include_drag=True, include_SRP=True, include_J3J4=True):
        self.Cd = Cd
        self.A_cross = A_cross
        self.mass = mass
        self.epoch = epoch if epoch is not None else EPOCH
        self.include_drag = include_drag
        self.include_SRP = include_SRP
        self.include_J3J4 = include_J3J4
        self._density_cache_time = -1e30
        self._cached_density = 0.0
        self._density_cache_interval = 60.0  # refresh every 60 s

    def _get_sun_position(self, t_since_epoch):
        """Approximate Sun position in ECI using simple analytical model.

        Parameters
        ----------
        t_since_epoch : float
            Seconds since mission epoch.

        Returns
        -------
        ndarray, shape (3,)
            Sun ECI position [m].
        """
        current_time = self.epoch + timedelta(seconds=t_since_epoch)
        doy = current_time.timetuple().tm_yday
        # Mean anomaly of Sun (approximate, epoch at vernal equinox ~day 80)
        M_sun = 2.0 * math.pi * (doy - 80) / 365.25
        obliquity = math.radians(23.4393)
        r_sun = AU * np.array([
            math.cos(M_sun),
            math.sin(M_sun) * math.cos(obliquity),
            math.sin(M_sun) * math.sin(obliquity)
        ])
        return r_sun

    def _get_cached_density(self, r, t):
        """Cache NRLMSISE-00 density calls (expensive).

        Parameters
        ----------
        r : ndarray, shape (3,)
            ECI position [m].
        t : float
            Seconds since epoch.

        Returns
        -------
        float
            Atmospheric density [kg/m^3].
        """
        if t - self._density_cache_time > self._density_cache_interval:
            self._density_cache_time = t
            self._cached_density = get_atmospheric_density(r, self.epoch, t)
        return self._cached_density

    def eom(self, t, state):
        """Equations of motion: dr/dt = v, dv/dt = a_total.

        Parameters
        ----------
        t : float
            Time since epoch [s].
        state : ndarray, shape (6,)
            [x, y, z, vx, vy, vz] in ECI [m, m/s].

        Returns
        -------
        ndarray, shape (6,)
            State derivative.
        """
        r = state[:3]
        v = state[3:6]
        R = np.linalg.norm(r)

        # Two-body
        a = -MU_EARTH / R**3 * r

        # J2
        a += accel_J2(r)

        # J3, J4
        if self.include_J3J4:
            a += accel_J3(r)
            a += accel_J4(r)

        # Drag
        if self.include_drag:
            rho = self._get_cached_density(r, t)
            a += accel_drag(r, v, self.Cd, self.A_cross, self.mass, rho)

        # SRP
        if self.include_SRP:
            r_sun = self._get_sun_position(t)
            a += accel_SRP(r, r_sun, self.A_cross, CR, self.mass)

        return np.concatenate([v, a])

    def propagate(self, state0, t_span, t_eval=None):
        """Propagate orbit from state0 over t_span.

        Parameters
        ----------
        state0 : ndarray, shape (6,)
            Initial state [x,y,z,vx,vy,vz] in ECI [m, m/s].
        t_span : tuple (t0, tf)
            Integration interval [s since epoch].
        t_eval : ndarray or None
            Times at which to store solution.

        Returns
        -------
        sol : OdeResult
            scipy solve_ivp solution object.
        """
        # Reset density cache at start of propagation
        self._density_cache_time = -1e30

        sol = solve_ivp(self.eom, t_span, state0, method='RK45',
                        rtol=1e-10, atol=1e-12, max_step=DT_INTEGRATOR,
                        t_eval=t_eval, dense_output=True)
        return sol

    def propagate_multiple_orbits(self, state0, n_orbits, n_points_per_orbit=360):
        """Convenience method: propagate for n_orbits.

        Parameters
        ----------
        state0 : ndarray, shape (6,)
            Initial ECI state.
        n_orbits : int
            Number of orbits.
        n_points_per_orbit : int
            Output points per orbit.

        Returns
        -------
        sol : OdeResult
            Solution object.
        """
        from cpfc_simulation.config import T_ORBIT
        t_end = n_orbits * T_ORBIT
        t_eval = np.linspace(0, t_end, n_orbits * n_points_per_orbit + 1)
        return self.propagate(state0, (0.0, t_end), t_eval=t_eval)
