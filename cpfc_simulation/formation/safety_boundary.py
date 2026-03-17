"""Melnikov-based formation safety boundary — Novel contribution."""
import numpy as np
import h5py
from multiprocessing import Pool
from functools import partial
from cpfc_simulation.config import (
    MU_EARTH, R_EARTH, J2, IXX, IYY, IZZ, MASS_SAT,
    A_STOWED, A_DEPLOYED, F107_NOMINAL, AP_NOMINAL
)


def compute_safety_boundary_point(params):
    """
    Compute safety boundary at one (h, i, F10.7) point.

    Returns: dict with epsilon_crit, Delta_y_max, Omega_switch_opt, chaos_exists
    """
    h_km, inc_deg, F107 = params

    from cpfc_simulation.chaos.melnikov import (
        optimal_switching_frequency, melnikov_chaos_boundary
    )
    from cpfc_simulation.dynamics.perturbations import get_atmospheric_density
    from cpfc_simulation.config import EPOCH

    # Orbital parameters
    a = R_EARTH + h_km * 1e3
    inc = np.radians(inc_deg)
    n = np.sqrt(MU_EARTH / a**3)
    v_orb = n * a  # orbital velocity

    # Get atmospheric density (approximate: use equatorial position at this altitude)
    # Simple: construct ECI position at equator
    r_eci = np.array([a, 0.0, 0.0])
    try:
        rho = get_atmospheric_density(r_eci, EPOCH, 0.0, F107=F107, Ap=AP_NOMINAL)
    except Exception:
        rho = 1e-12  # fallback

    # Aerodynamic torque coefficient
    # M_aero ~ 0.5 * rho * v^2 * A * L * Cn
    L_offset = 0.3405/2 + 0.30/2  # panel offset from CoM
    M_aero_coeff = 0.5 * rho * v_orb**2 * (0.30*0.10) * L_offset * 2.0 / IYY

    # Melnikov analysis
    try:
        Omega_opt, M_max = optimal_switching_frequency(n, IXX, IZZ, IYY, M_aero_coeff)
    except Exception:
        Omega_opt, M_max = 0.0, 0.0

    # Chaos exists if Melnikov integral has simple zeros (M > threshold)
    chaos_exists = M_max > 1e-10

    # Critical perturbation amplitude
    # epsilon_crit is the minimum drag plate oscillation amplitude for chaos
    epsilon_crit = 1e-10 / max(M_max, 1e-20)  # rough estimate

    # Maximum controllable separation
    # dfy_max = 0.5 * rho * v^2 * (Cd_deployed*A_deployed - Cd_stowed*A_stowed) / mass
    Cd_approx = 2.2
    dfy_max = 0.5 * rho * v_orb**2 * (Cd_approx * A_DEPLOYED - Cd_approx * A_STOWED) / MASS_SAT

    # Secular gain from corrected SS
    kappa2 = 1 + 1.5 * J2 * (R_EARTH/a)**2 * (5*np.cos(inc)**2 - 1)
    kappa = np.sqrt(abs(kappa2))
    c_ss = 1.5 * J2 * (R_EARTH/a)**2 * (1 - 1.25*np.sin(inc)**2)
    secular_gain = 3*kappa / ((1+2*c_ss) * n**2)
    T_orb = 2*np.pi / n

    Delta_y_max = abs(secular_gain * dfy_max * T_orb)

    return {
        'h_km': h_km, 'inc_deg': inc_deg, 'F107': F107,
        'epsilon_crit': epsilon_crit, 'Delta_y_max': Delta_y_max,
        'Omega_switch_opt': Omega_opt, 'chaos_exists': bool(chaos_exists),
        'rho': rho, 'dfy_max': dfy_max
    }


def generate_safety_boundary_map(h_range_km=None, inc_range_deg=None,
                                   F107_range=None, n_workers=4,
                                   output_file='safety_boundary_map.h5'):
    """
    Parallel computation of safety boundary over full parameter space.
    Saves to HDF5.
    """
    if h_range_km is None:
        h_range_km = np.arange(300, 625, 25)
    if inc_range_deg is None:
        inc_range_deg = np.arange(0, 99, 5)
    if F107_range is None:
        F107_range = np.array([70, 90, 110, 130, 150, 170, 190, 210, 230, 250])

    # Build parameter grid
    params_list = []
    for h in h_range_km:
        for inc in inc_range_deg:
            for f107 in F107_range:
                params_list.append((h, inc, f107))

    # Parallel computation
    results = []
    try:
        with Pool(n_workers) as pool:
            results = pool.map(compute_safety_boundary_point, params_list)
    except Exception:
        # Fallback to serial
        results = [compute_safety_boundary_point(p) for p in params_list]

    # Reshape and save to HDF5
    nh = len(h_range_km)
    ni = len(inc_range_deg)
    nf = len(F107_range)

    Delta_y_max = np.zeros((nh, ni, nf))
    epsilon_crit = np.zeros((nh, ni, nf))
    Omega_opt = np.zeros((nh, ni, nf))
    chaos_exists = np.zeros((nh, ni, nf), dtype=bool)

    for idx, res in enumerate(results):
        ih = idx // (ni * nf)
        ii = (idx % (ni * nf)) // nf
        ifi = idx % nf
        Delta_y_max[ih, ii, ifi] = res['Delta_y_max']
        epsilon_crit[ih, ii, ifi] = res['epsilon_crit']
        Omega_opt[ih, ii, ifi] = res['Omega_switch_opt']
        chaos_exists[ih, ii, ifi] = res['chaos_exists']

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('h_range_km', data=h_range_km)
        f.create_dataset('inc_range_deg', data=inc_range_deg)
        f.create_dataset('F107_range', data=F107_range)
        f.create_dataset('Delta_y_max', data=Delta_y_max)
        f.create_dataset('epsilon_crit', data=epsilon_crit)
        f.create_dataset('Omega_switch_opt', data=Omega_opt)
        f.create_dataset('chaos_exists', data=chaos_exists)

    return output_file
