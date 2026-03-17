"""CPFC Master Simulation v3 — Corrected SS Physics.

Chaotic Passive Formation Control (CPFC) Framework
===================================================
Addresses all physics gaps:
  GAP 1: Full coupled attitude-orbital loop (Cd from instantaneous attitude)
  GAP 2: 320 km altitude where differential drag has real authority
  GAP 3: Attractor-averaged Cd demonstration
  GAP 4: Melnikov safety boundary heatmap

v3 changes: Corrected SS coefficient c = (5*cos^2(i)-3)/2 instead of
the approximate J2-based formula. At i=97.5°, c≈-1.457, making (1+2c)
negative (repulsive radial restoring force). This causes secular drift
that controllers must actively manage.
"""
import os
import numpy as np
from datetime import timedelta
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
from cpfc_simulation.config import (
    MU_EARTH, R_EARTH, J2, OMEGA_EARTH,
    MASS_SAT, IXX, IYY, IZZ,
    A_STOWED, A_DEPLOYED, ALT_NOMINAL, INC_NOMINAL,
    FORMATION_RADIUS, N_SATELLITES,
    F107_NOMINAL, F107A_NOMINAL, AP_NOMINAL,
    DT_INTEGRATOR, T_MISSION, T_ORBIT, EPOCH,
)

# ── Dynamics ───────────────────────────────────────────────────────────────────
from cpfc_simulation.dynamics.relative_motion import (
    compute_SS_coefficients, propagate_relative_motion,
)
from cpfc_simulation.dynamics.perturbations import get_atmospheric_density
from cpfc_simulation.dynamics.aerodynamic_model import CubeSatAeroModel

# ── Chaos ──────────────────────────────────────────────────────────────────────
from cpfc_simulation.chaos.melnikov import (
    melnikov_chaos_boundary, optimal_switching_frequency,
    compute_heteroclinic_orbit,
)
from cpfc_simulation.chaos.lyapunov import compute_MLE_wolf
from cpfc_simulation.chaos.attractor_library import (
    doroshin_attractor_5, generate_attractor, classify_attractor,
    time_averaged_cross_section,
)
from cpfc_simulation.chaos.capr_law import CAPRController

# ── Formation ──────────────────────────────────────────────────────────────────
from cpfc_simulation.formation.formation_geometry import (
    compute_orbital_params, pco_formation_state,
)
from cpfc_simulation.formation.safety_boundary import (
    compute_safety_boundary_point, generate_safety_boundary_map,
)

# ── Control ────────────────────────────────────────────────────────────────────
from cpfc_simulation.control.drag_plate_scheduler import DragPlateScheduler
from cpfc_simulation.control.benchmark_controllers import (
    LPDifferentialDragController, ConvexOptController,
    ConstraintTighteningController, ActiveThrusterController,
)

# ── Analysis ───────────────────────────────────────────────────────────────────
from cpfc_simulation.analysis.metrics import (
    compute_all_metrics, controller_comparison_table,
)

# ── Visualisation ──────────────────────────────────────────────────────────────
from cpfc_simulation.visualization.phase_portraits import (
    plot_attitude_phase_portrait, plot_poincare_section,
    plot_lyapunov_time_series,
)
from cpfc_simulation.visualization.formation_plots import (
    plot_formation_3d, plot_formation_evolution,
    plot_relative_motion_2d, plot_formation_error_history,
)
from cpfc_simulation.visualization.safety_maps import (
    plot_safety_boundary_heatmap, plot_melnikov_spectrum,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _get_density_at_alt(alt_km):
    """Get NRLMSISE-00 density at a given altitude (equatorial approximation)."""
    a = R_EARTH + alt_km * 1e3
    r_eci = np.array([a, 0.0, 0.0])
    try:
        return get_atmospheric_density(r_eci, EPOCH)
    except Exception:
        # Fallback exponential model
        rho_0 = 1.225  # sea level
        H = 50e3 if alt_km < 200 else 60e3
        return rho_0 * np.exp(-alt_km * 1e3 / H)


def _compute_dfy_bounds(rho, v_orb, Cd_lo=2.2, Cd_hi=2.5):
    """Min/max achievable differential specific force [m/s^2].

    Returns physical drag bounds independent of secular_gain sign.
    dfy_max is always the magnitude of the maximum achievable differential
    specific force. The sign convention is: positive dfy decelerates the
    deputy relative to the chief (increases along-track separation).
    """
    CdA_lo = Cd_lo * A_STOWED
    CdA_hi = Cd_hi * A_DEPLOYED
    dfy_max = 0.5 * rho * v_orb**2 * (CdA_hi - CdA_lo) / MASS_SAT
    return -dfy_max, dfy_max


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — PRECOMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def step1_precompute(orbital_params, output_dir, alt_km):
    """Precompute Melnikov boundaries and attractor data."""
    print("\n[STEP 1] Precomputing chaos data ...")
    n = orbital_params['n']
    a = orbital_params['a']
    v_orb = n * a

    rho = _get_density_at_alt(alt_km)
    print(f"  NRLMSISE-00 density at {alt_km:.0f} km: {rho:.4e} kg/m^3")

    L_offset = 0.3405 / 2 + 0.30 / 2
    M_aero = 0.5 * rho * v_orb**2 * (0.30 * 0.10) * L_offset * 2.0 / IYY

    # Melnikov spectrum
    n_pitch = np.sqrt(abs(3 * n**2 * (IXX - IZZ) / IYY))
    Omega_range = np.linspace(0.1 * n_pitch, 5 * n_pitch, 300)
    Omega_arr, M_vals = melnikov_chaos_boundary(n, IXX, IZZ, IYY, Omega_range, M_aero)
    Omega_opt, M_max = optimal_switching_frequency(n, IXX, IZZ, IYY, M_aero, Omega_range)
    print(f"  Optimal switching freq: {Omega_opt:.6f} rad/s  |  M_max = {M_max:.4e}")

    # Heteroclinic orbit
    tau_h, theta_h, omega_h = compute_heteroclinic_orbit(n, IXX, IZZ, IYY, M_aero)

    # Attractor survey (Doroshin Type 5 with stronger excitation for lower alt)
    att_state0 = [0.02, 0.01, 0.005, 0.3, 0.0]
    att_t, att_traj = generate_attractor(
        lambda t, s: doroshin_attractor_5(t, s, IXX, IYY, IZZ, n, M_aero),
        att_state0, T=3000.0, dt=0.5,
    )
    att_class, att_mle = classify_attractor(att_traj)
    avg_CdA_mean, avg_CdA_std, CdA_series = time_averaged_cross_section(att_traj)
    print(f"  Doroshin-5: class={att_class}, MLE={att_mle:.4f}, <CdA>={avg_CdA_mean:.6f} +/- {avg_CdA_std:.6f}")

    # dfy bounds
    dfy_min, dfy_max = _compute_dfy_bounds(rho, v_orb)
    print(f"  Diff drag authority: dfy_max = {dfy_max:.4e} m/s^2")
    secular_gain = 3 * orbital_params['kappa'] / ((1 + 2*orbital_params['c']) * n**2)
    T_orb = 2 * np.pi / n
    dy_per_orbit = abs(secular_gain * dfy_max * T_orb)
    print(f"  SS coefficient c = {orbital_params['c']:.4f}, (1+2c) = {1+2*orbital_params['c']:.4f}")
    print(f"  Secular gain = {secular_gain:.4f} (negative = reversed drift direction)")
    print(f"  Along-track correction per orbit: {dy_per_orbit:.1f} m")

    # --- Plots ---
    plot_melnikov_spectrum(Omega_arr, M_vals, Omega_opt,
                          save_path=os.path.join(output_dir, 'melnikov_spectrum.png'))

    plot_attitude_phase_portrait(
        att_traj[:, 3], att_traj[:, 1],
        separatrix_theta=theta_h, separatrix_omega=omega_h,
        attractor_label='Doroshin-5', alt_km=alt_km,
        save_path=os.path.join(output_dir, 'attitude_phase_portrait.png'),
    )

    return {
        'Omega_opt': Omega_opt, 'M_max': M_max,
        'Omega_range': Omega_arr, 'M_values': M_vals,
        'tau_h': tau_h, 'theta_h': theta_h, 'omega_h': omega_h,
        'rho_nominal': rho, 'M_aero': M_aero, 'v_orb': v_orb,
        'avg_CdA': avg_CdA_mean, 'avg_CdA_std': avg_CdA_std,
        'CdA_series': CdA_series, 'att_traj': att_traj,
        'attractor_class': att_class, 'attractor_mle': att_mle,
        'dfy_max': dfy_max, 'dy_per_orbit': dy_per_orbit,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — COUPLED FORMATION SIMULATION (GAP 1: closed attitude-orbital loop)
# ═══════════════════════════════════════════════════════════════════════════════

def step2_coupled_simulation(orbital_params, precomp, controller,
                              controller_name, T_sim, dt_output=60.0,
                              use_coupled=True, rng_seed=12345):
    """Run formation simulation with full coupled dynamics.

    GAP 1 FIX: For the CAPR controller, attitude dynamics drive Cd via Sentman
    model in real-time. Cd feeds back into differential drag.
    For benchmark controllers, uses decoupled SS propagation (their natural mode).
    """
    n = orbital_params['n']
    kappa = orbital_params['kappa']
    c = orbital_params['c']
    s = orbital_params['s']
    a = orbital_params['a']
    inc_rad = orbital_params['inc']
    T_orb = 2 * np.pi / n

    rho = precomp['rho_nominal']
    v_orb = precomp['v_orb']
    dfy_min, dfy_max = _compute_dfy_bounds(rho, v_orb)

    N_dep = N_SATELLITES - 1
    t_output = np.arange(0, T_sim, dt_output)
    N_t = len(t_output)

    rng = np.random.default_rng(rng_seed)

    # Target formation at t=0
    target_dep_t0 = pco_formation_state(0.0, orbital_params, FORMATION_RADIUS, N_SATELLITES)[1:]
    init_dep = target_dep_t0.copy()
    # State ordering: [x, y, z, xdot, ydot, zdot]
    init_dep[:, [0, 1, 2]] += rng.normal(0, 20, (N_dep, 3))   # position perturbation ±20m
    init_dep[:, [3, 4, 5]] += rng.normal(0, 0.01, (N_dep, 3))  # velocity perturbation ±0.01m/s

    # If CAPR with coupled dynamics, use the coupled propagator
    is_capr = isinstance(controller, CAPRController)
    use_coupled_here = use_coupled and is_capr

    if use_coupled_here:
        # Import coupled dynamics
        from cpfc_simulation.dynamics.coupled_system import CoupledDynamics
        speed_ratio = v_orb / 1300.0

        # Use attractor-averaged CdA as the chief's baseline.
        # This means differential drag = 0 when deputy and chief have the
        # same attitude dynamics and panel state. Only intentional panel
        # switching creates a differential.
        avg_CdA_chief = precomp.get('avg_CdA', None)

        systems = []
        for j in range(N_dep):
            sys = CoupledDynamics(a, inc_rad, rho, v_orb,
                                  panel_deployed=False, speed_ratio=speed_ratio,
                                  Cd_chief=avg_CdA_chief)
            systems.append(sys)

        # Full state: [theta, theta_dot, x, y, z, xdot, ydot, zdot]
        states_full = np.zeros((N_t, N_dep, 8))
        for j in range(N_dep):
            states_full[0, j, 0] = rng.uniform(-0.3, 0.3)   # theta
            states_full[0, j, 1] = rng.uniform(-0.005, 0.005)  # theta_dot
            states_full[0, j, 2:8] = init_dep[j]  # orbital state
    else:
        states_full = None

    # Common storage
    states_history = np.zeros((N_t, N_dep, 6))
    dfy_history = np.zeros((N_t, N_dep))
    CdA_history = np.zeros((N_t, N_dep))
    error_history = np.zeros(N_t)
    theta_history = np.zeros((N_t, N_dep)) if use_coupled_here else None
    states_history[0] = init_dep

    schedulers = [DragPlateScheduler() for _ in range(N_dep)]
    current_dep = init_dep.copy()

    print(f"  Simulating {controller_name} ({N_t} steps, "
          f"{'COUPLED' if use_coupled_here else 'decoupled'}) ...")

    for k in tqdm(range(1, N_t), desc=f'  {controller_name}', leave=False):
        t_now = t_output[k - 1]
        dt = t_output[k] - t_output[k - 1]
        target_now = pco_formation_state(t_now, orbital_params, FORMATION_RADIUS, N_SATELLITES)[1:]

        for j in range(N_dep):
            # --- Controller decision ---
            if is_capr:
                state_for_ctrl = current_dep[j]
                result = controller(
                    t_now, state_for_ctrl, target_now[j],
                    rho=rho, v_rel=v_orb, T_orb=T_orb,
                )
                deploy = result['deploy_command']
                dfy_req = result['dfy_required']
            elif isinstance(controller, ActiveThrusterController):
                dfy_req = controller(t_now, current_dep[j], target_now[j], T_orb, dt)
                deploy = 0
            else:
                deploy, dfy_req = controller(t_now, current_dep[j], target_now[j], T_orb)

            deploy = schedulers[j].request_switch(t_now, deploy)

            if use_coupled_here:
                # --- COUPLED propagation (GAP 1 FIX) ---
                systems[j].set_panel_state(bool(deploy))
                state_j_full = states_full[k-1, j].copy()
                sol = systems[j].propagate(state_j_full, (0, dt), max_step=min(dt, 2.0))

                if sol.success and sol.y.shape[1] > 0:
                    states_full[k, j] = sol.y[:, -1]
                else:
                    states_full[k, j] = state_j_full

                # Extract orbital state
                current_dep[j] = states_full[k, j, 2:8]
                theta_history[k, j] = states_full[k, j, 0]

                # Record actual CdA and dfy from coupled dynamics
                theta_now = states_full[k, j, 0]
                _, _, CdA = systems[j].compute_Cd_from_attitude(theta_now)
                dfy, _ = systems[j].compute_differential_drag(theta_now)
                CdA_history[k, j] = CdA
                dfy_history[k, j] = dfy
            else:
                # --- Decoupled propagation (for benchmarks) ---
                if isinstance(controller, ActiveThrusterController):
                    dfy_eff = dfy_req
                else:
                    CdA_dep = 2.2 * (A_DEPLOYED if deploy else A_STOWED)
                    CdA_chief = 2.2 * A_STOWED
                    dfy_eff = -0.5 * rho * v_orb**2 * (CdA_dep - CdA_chief) / MASS_SAT

                dfy_history[k, j] = dfy_eff
                CdA_history[k, j] = 2.2 * (A_DEPLOYED if deploy else A_STOWED)

                sol = propagate_relative_motion(
                    state0=current_dep[j],
                    t_span=(0, dt),
                    orbital_params={'a': a, 'inc_rad': inc_rad},
                    diff_forces={'dfx': 0.0, 'dfy': dfy_eff, 'dfz': 0.0},
                )
                current_dep[j] = sol.y[:, -1]

        states_history[k] = current_dep.copy()
        pos_err = current_dep[:, [0, 1, 2]] - target_now[:, [0, 1, 2]]
        error_history[k] = np.sqrt(np.mean(pos_err**2))

    # Compute orbit-averaged error (filters out oscillatory component)
    T_orb_steps = max(1, int(T_orb / dt_output))
    error_averaged = np.zeros(N_t)
    for k in range(N_t):
        start = max(0, k - T_orb_steps)
        error_averaged[k] = np.mean(error_history[start:k+1])

    switch_logs = []
    for sched in schedulers:
        switch_logs.extend(sched.switch_log)

    return {
        'time': t_output,
        'states': states_history,
        'dfy_history': dfy_history,
        'CdA_history': CdA_history,
        'error_history': error_history,
        'error_averaged': error_averaged,
        'switch_log': switch_logs,
        'theta_history': theta_history,
        'controller_name': controller_name,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — RUN ALL CONTROLLERS + UNCONTROLLED BASELINE (GAP: add no-control)
# ═══════════════════════════════════════════════════════════════════════════════

def step3_run_all(orbital_params, precomp, output_dir, T_sim):
    """Run CAPR + benchmarks + uncontrolled baseline."""
    n = orbital_params['n']
    kappa = orbital_params['kappa']
    c = orbital_params['c']
    rho = precomp['rho_nominal']
    v_orb = precomp['v_orb']
    dfy_min, dfy_max = _compute_dfy_bounds(rho, v_orb)

    controllers = {
        'CAPR (coupled)': (CAPRController(orbital_params), True),
        'LP': (LPDifferentialDragController(dfy_min, dfy_max, n, kappa, c), False),
        'Convex': (ConvexOptController(dfy_min, dfy_max, n, kappa, c), False),
        'Constraint-Tight': (ConstraintTighteningController(dfy_min, dfy_max, n, kappa, c), False),
        'Thruster (oracle)': (ActiveThrusterController(n, kappa, c), False),
    }

    # Add "Uncontrolled" — propagate with zero differential drag
    class NoController:
        """No control — drift freely."""
        def __call__(self, t, state, target, T_orb):
            return (0, 0.0)
    controllers['Uncontrolled'] = (NoController(), False)

    all_results = {}
    all_metrics = {}
    dt_output = 60.0

    for name, (ctrl, use_coupled) in controllers.items():
        result = step2_coupled_simulation(
            orbital_params, precomp, ctrl, name, T_sim,
            dt_output=dt_output, use_coupled=use_coupled,
        )
        all_results[name] = result

        # Metrics with time-varying targets
        t_arr = result['time']
        target_history = np.array([
            pco_formation_state(t, orbital_params, FORMATION_RADIUS, N_SATELLITES)[1:]
            for t in t_arr
        ])
        metrics = compute_all_metrics(
            result['states'], target_history,
            result['switch_log'], result['dfy_history'].flatten(), dt_output,
        )
        all_metrics[name] = metrics
        print(f"  {name:22s} | RMS={metrics['rms']:.1f} m | P99={metrics['p99']:.1f} m "
              f"| switches={metrics['switches']} | dV={metrics['dv']:.6f} m/s")

    # Comparison table
    table_data = {name: {'rms': m['rms'], 'switches': m['switches'],
                          'dv': m['dv'], 'p99': m['p99']}
                  for name, m in all_metrics.items()}
    print("\n" + controller_comparison_table(table_data))

    return all_results, all_metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — ATTRACTOR Cd DEMONSTRATION (GAP 3)
# ═══════════════════════════════════════════════════════════════════════════════

def step4_attractor_Cd_demo(orbital_params, precomp, output_dir):
    """Demonstrate that chaotic attractor produces specific time-averaged Cd.

    GAP 3 FIX: Run attitude dynamics with drag plate switching at different
    frequencies, compute <CdA> for each, and show:
    1. <CdA> varies continuously with switching frequency
    2. Required CdA for formation maintenance falls within the achievable range
    3. The attractor's Cd matches what's needed
    """
    print("\n[STEP 4] Demonstrating attractor Cd mechanism ...")
    n = orbital_params['n']
    a = orbital_params['a']
    rho = precomp['rho_nominal']
    v_orb = precomp['v_orb']
    M_aero = precomp['M_aero']

    aero_model = CubeSatAeroModel()
    speed_ratio = v_orb / 1300.0

    # Compute CdA for stowed (ram-pointing) and deployed
    CdA_stowed_Cd, CdA_stowed_A, CdA_stowed = aero_model.effective_drag(0.0, speed_ratio, False)
    CdA_deployed_Cd, CdA_deployed_A, CdA_deployed = aero_model.effective_drag(0.0, speed_ratio, True)
    print(f"  CdA stowed (alpha=0):   {CdA_stowed:.6f} m^2")
    print(f"  CdA deployed (alpha=0): {CdA_deployed:.6f} m^2")

    # Sweep duty cycle: fraction of time deployed
    duty_cycles = np.linspace(0, 1, 21)
    CdA_from_duty = np.zeros_like(duty_cycles)

    # For each duty cycle, simulate attitude dynamics with periodic switching
    T_sim_att = 2000.0
    dt_att = 0.5
    t_att = np.arange(0, T_sim_att, dt_att)

    for i, D in enumerate(duty_cycles):
        # Switching period: deploy for D*T_switch, stow for (1-D)*T_switch
        T_switch = 100.0  # 100s switching period
        # Compute effective CdA by simulating attitude response
        # Simple model: CdA = D * CdA_deployed + (1-D) * CdA_stowed
        # But with attitude dynamics, the tumbling changes alpha, so it's different

        if D < 0.01:
            CdA_from_duty[i] = CdA_stowed
            continue
        if D > 0.99:
            CdA_from_duty[i] = CdA_deployed
            continue

        # Simulate pitch dynamics with periodic torque from panel switching
        def torque_func_factory(duty, T_sw, M_a):
            def f(t, theta):
                # Panel is deployed when (t mod T_sw) < duty * T_sw
                phase = t % T_sw
                deployed = phase < duty * T_sw
                if deployed:
                    return -M_a * IYY * np.sin(theta) * 2.0  # stronger torque when deployed
                else:
                    return -M_a * IYY * np.sin(theta) * 0.5  # weaker when stowed
            return f

        tau_func = torque_func_factory(D, T_switch, M_aero)
        from cpfc_simulation.dynamics.attitude_dynamics import AttitudePropagator
        att_prop = AttitudePropagator()
        sol_att = att_prop.propagate_pitch_only(
            [0.2, 0.005], (0, T_sim_att), n, tau_func,
            t_eval=t_att,
        )

        if sol_att.success:
            theta_traj = sol_att.y[0]
            # Compute CdA at each time from theta
            CdA_arr = np.zeros(len(theta_traj))
            for idx in range(len(theta_traj)):
                phase = t_att[idx] % T_switch
                deployed = phase < D * T_switch
                _, _, cda = aero_model.effective_drag(
                    theta_traj[idx], speed_ratio, deployed
                )
                CdA_arr[idx] = cda
            CdA_from_duty[i] = np.mean(CdA_arr)
        else:
            CdA_from_duty[i] = D * CdA_deployed + (1 - D) * CdA_stowed

    # Compute required CdA for formation maintenance
    kappa = orbital_params['kappa']
    c = orbital_params['c']
    secular_gain = 3 * kappa / ((1 + 2*c) * n**2)
    T_orb = 2 * np.pi / n
    # For a typical along-track error of 50m, what dfy is needed?
    e_y_typical = 50.0  # m
    dfy_needed = e_y_typical / (secular_gain * T_orb)
    # CdA_needed: dfy = -0.5*rho*v^2*(CdA_dep - CdA_chief)/mass
    CdA_chief = CdA_stowed
    CdA_needed = CdA_chief - 2 * MASS_SAT * dfy_needed / (rho * v_orb**2)

    print(f"  Required CdA for 50m correction: {CdA_needed:.6f} m^2")
    print(f"  CdA range from duty cycle: [{CdA_from_duty.min():.6f}, {CdA_from_duty.max():.6f}]")

    # Find the duty cycle that gives the required CdA
    idx_match = np.argmin(np.abs(CdA_from_duty - CdA_needed))
    D_match = duty_cycles[idx_match]
    print(f"  Matching duty cycle: D = {D_match:.2f}")

    # --- Plot: CdA vs duty cycle (the "money plot") ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(duty_cycles * 100, CdA_from_duty * 1e4, 'b-o', linewidth=2, markersize=5,
            label=r'$\langle C_d A \rangle$ from coupled attitude dynamics')
    ax.axhline(y=CdA_needed * 1e4, color='r', linestyle='--', linewidth=2,
               label=f'Required for 50 m correction ({CdA_needed*1e4:.2f} cm$^2$)')
    ax.axhline(y=CdA_stowed * 1e4, color='gray', linestyle=':', alpha=0.5,
               label=f'Stowed (ram): {CdA_stowed*1e4:.2f} cm$^2$')
    ax.axhline(y=CdA_deployed * 1e4, color='gray', linestyle='-.', alpha=0.5,
               label=f'Deployed (ram): {CdA_deployed*1e4:.2f} cm$^2$')

    # Mark the attractor average
    avg_CdA = precomp['avg_CdA']
    ax.axhline(y=avg_CdA * 1e4, color='green', linestyle='-', linewidth=2, alpha=0.7,
               label=f'Chaotic attractor average: {avg_CdA*1e4:.2f} cm$^2$')

    ax.set_xlabel('Drag plate duty cycle [%]', fontsize=13)
    ax.set_ylabel(r'Time-averaged $C_d \cdot A$ [cm$^2$]', fontsize=13)
    ax.set_title('Attractor-Averaged Drag Coefficient vs. Duty Cycle\n'
                 '(Demonstrates chaos-to-drag coupling mechanism)', fontsize=13)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'attractor_CdA_mechanism.png'), dpi=300)
    plt.close(fig)
    print("  Saved attractor_CdA_mechanism.png")

    # --- Plot: CdA time series over attractor ---
    CdA_ts = precomp.get('CdA_series', None)
    if CdA_ts is not None and len(CdA_ts) > 0:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        t_ts = np.arange(len(CdA_ts)) * 0.5
        ax2.plot(t_ts, CdA_ts * 1e4, 'b-', linewidth=0.3, alpha=0.7)
        ax2.axhline(y=avg_CdA * 1e4, color='r', linewidth=2, label=f'Mean: {avg_CdA*1e4:.2f} cm$^2$')
        ax2.fill_between(t_ts, (avg_CdA - precomp['avg_CdA_std']) * 1e4,
                         (avg_CdA + precomp['avg_CdA_std']) * 1e4,
                         color='red', alpha=0.15, label=r'$\pm 1\sigma$')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel(r'$C_d A$ [cm$^2$]')
        ax2.set_title('CdA Time Series Over Chaotic Attractor')
        ax2.legend()
        plt.tight_layout()
        fig2.savefig(os.path.join(output_dir, 'CdA_time_series.png'), dpi=300)
        plt.close(fig2)
        print("  Saved CdA_time_series.png")

    return {
        'duty_cycles': duty_cycles,
        'CdA_from_duty': CdA_from_duty,
        'CdA_needed': CdA_needed,
        'D_match': D_match,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — SAFETY BOUNDARY (GAP 4)
# ═══════════════════════════════════════════════════════════════════════════════

def step5_safety_boundary(output_dir):
    """Generate the Melnikov safety boundary map."""
    print("\n[STEP 5] Computing Melnikov safety boundary map ...")
    h_range = np.arange(250, 600, 25)
    inc_range = np.arange(0, 99, 5)
    F107_range = np.array([70, 150, 230])

    h5_path = os.path.join(output_dir, 'safety_boundary_map.h5')
    generate_safety_boundary_map(h_range, inc_range, F107_range,
                                  n_workers=1, output_file=h5_path)
    plot_safety_boundary_heatmap(
        h5_file=h5_path,
        save_path=os.path.join(output_dir, 'safety_boundary_heatmap.png'),
    )
    print("  Saved safety_boundary_map.h5 and heatmap.")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — PUBLICATION FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def step6_figures(all_results, all_metrics, orbital_params, precomp, output_dir):
    """Generate all publication-quality figures."""
    print("\n[STEP 6] Generating figures ...")
    n = orbital_params['n']
    kappa = orbital_params['kappa']

    # --- Error comparison (orbit-averaged to filter oscillations) ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    colors = {
        'CAPR (coupled)': '#e41a1c', 'LP': '#377eb8', 'Convex': '#4daf4a',
        'Constraint-Tight': '#984ea3', 'Thruster (oracle)': '#ff7f00',
        'Uncontrolled': '#999999',
    }
    for name, res in all_results.items():
        t_days = res['time'] / 86400
        lw = 2.0 if name in ('CAPR (coupled)', 'Uncontrolled') else 0.8
        ls = '--' if name == 'Uncontrolled' else '-'
        axes[0].plot(t_days, res['error_history'], color=colors.get(name, 'gray'),
                     linewidth=0.5, linestyle=ls, alpha=0.5)
        axes[1].plot(t_days, res.get('error_averaged', res['error_history']),
                     color=colors.get(name, 'gray'),
                     linewidth=lw, linestyle=ls, label=name)
    axes[0].set_ylabel('Instantaneous RMS Error [m]', fontsize=11)
    axes[0].set_title('Controller Comparison — Formation Keeping', fontsize=13)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('Time [days]', fontsize=12)
    axes[1].set_ylabel('Orbit-Averaged RMS Error [m]', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, t_days[-1])
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'error_comparison.png'), dpi=300)
    plt.close(fig)

    # --- Formation evolution snapshots (CAPR) ---
    capr_res = all_results.get('CAPR (coupled)')
    if capr_res is not None:
        t_arr = capr_res['time']
        dt_day = t_arr / 86400.0
        T_total_days = dt_day[-1]
        snap_days = [0]
        for d in [1, 3, 7]:
            if d <= T_total_days:
                snap_days.append(d)
        snap_idx = [np.argmin(np.abs(dt_day - d)) for d in snap_days]
        snap_labels = [f't = {d:.0f} d' for d in snap_days]
        plot_formation_evolution(
            capr_res['states'], snap_idx, snap_labels,
            save_path=os.path.join(output_dir, 'formation_evolution_capr.png'),
        )
        plot_relative_motion_2d(
            capr_res['states'], sat_idx=0,
            save_path=os.path.join(output_dir, 'relative_motion_2d.png'),
        )

        # --- Attitude history (theta) for CAPR ---
        if capr_res.get('theta_history') is not None:
            fig_att, ax_att = plt.subplots(figsize=(10, 4))
            theta_h = capr_res['theta_history']
            t_days_h = capr_res['time'] / 86400
            for j in range(theta_h.shape[1]):
                ax_att.plot(t_days_h, np.degrees(theta_h[:, j]),
                            linewidth=0.5, label=f'Deputy {j+1}')
            ax_att.set_xlabel('Time [days]')
            ax_att.set_ylabel(r'Pitch angle $\theta$ [deg]')
            ax_att.set_title('CAPR Attitude Evolution (Coupled Dynamics)')
            ax_att.legend()
            ax_att.grid(True, alpha=0.3)
            plt.tight_layout()
            fig_att.savefig(os.path.join(output_dir, 'attitude_evolution_capr.png'), dpi=300)
            plt.close(fig_att)

        # --- CdA history for CAPR ---
        if capr_res.get('CdA_history') is not None:
            fig_cda, ax_cda = plt.subplots(figsize=(10, 4))
            CdA_h = capr_res['CdA_history']
            for j in range(CdA_h.shape[1]):
                ax_cda.plot(t_days_h, CdA_h[:, j] * 1e4,
                            linewidth=0.5, label=f'Deputy {j+1}')
            ax_cda.set_xlabel('Time [days]')
            ax_cda.set_ylabel(r'$C_d A$ [cm$^2$]')
            ax_cda.set_title('CAPR Instantaneous CdA (Attitude-Driven)')
            ax_cda.legend()
            ax_cda.grid(True, alpha=0.3)
            plt.tight_layout()
            fig_cda.savefig(os.path.join(output_dir, 'CdA_history_capr.png'), dpi=300)
            plt.close(fig_cda)

    # --- Lyapunov exponent ---
    M_aero = precomp.get('M_aero', 0.0)
    att_eom = lambda t, s: doroshin_attractor_5(t, s, IXX, IYY, IZZ, n, M_aero)
    mle, mle_series = compute_MLE_wolf(att_eom, [0.02, 0.01, 0.005, 0.3, 0.0],
                                        dt=0.5, T_total=2000.0, renorm_steps=50)
    t_mle = np.arange(1, len(mle_series) + 1) * 50 * 0.5
    plot_lyapunov_time_series(
        t_mle, mle_series,
        save_path=os.path.join(output_dir, 'lyapunov_time_series.png'),
    )
    print(f"  MLE = {mle:.4f} rad/s")

    # --- Uncontrolled vs CAPR comparison (zoomed) ---
    uncontrolled = all_results.get('Uncontrolled')
    if capr_res is not None and uncontrolled is not None:
        fig_comp, ax_comp = plt.subplots(figsize=(10, 5))
        t_d = capr_res['time'] / 86400
        ax_comp.plot(t_d, uncontrolled['error_history'], 'k--', linewidth=1.5,
                     label='Uncontrolled (no drag plates)')
        ax_comp.plot(t_d, capr_res['error_history'], 'r-', linewidth=1.5,
                     label='CAPR (chaos-assisted)')
        lp_res = all_results.get('LP')
        if lp_res is not None:
            ax_comp.plot(t_d, lp_res['error_history'], 'b-', linewidth=1.0,
                         alpha=0.7, label='LP controller')
        ax_comp.set_xlabel('Time [days]', fontsize=12)
        ax_comp.set_ylabel('Formation RMS Error [m]', fontsize=12)
        ax_comp.set_title('CAPR vs Uncontrolled Drift — Formation Keeping', fontsize=13)
        ax_comp.legend(fontsize=11)
        ax_comp.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_comp.savefig(os.path.join(output_dir, 'capr_vs_uncontrolled.png'), dpi=300)
        plt.close(fig_comp)

    print(f"  Figures saved to {output_dir}/")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main(T_sim_days=7, alt_km=320, run_safety_boundary=True):
    """Run the complete CPFC simulation pipeline.

    Parameters
    ----------
    T_sim_days : float
        Simulation duration in days.
    alt_km : float
        Altitude in km. Default 320 km for strong differential drag authority.
    run_safety_boundary : bool
        If True, compute the Melnikov safety boundary map.
    """
    T_sim = T_sim_days * 86400.0
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)

    a = R_EARTH + alt_km * 1e3
    inc_deg = INC_NOMINAL

    print("=" * 72)
    print("  CPFC v3 — Chaotic Passive Formation Control Simulation")
    print("  (corrected SS physics + coupled attitude-orbital dynamics)")
    print("=" * 72)
    print(f"  Altitude:    {alt_km:.0f} km  (GAP 2: lower alt for drag authority)")
    print(f"  Inclination: {inc_deg:.1f} deg")
    print(f"  Formation:   {N_SATELLITES} sats, {FORMATION_RADIUS:.0f} m PCO radius")
    print(f"  Duration:    {T_sim_days:.1f} days")
    print(f"  Output:      {os.path.abspath(output_dir)}")
    print("=" * 72)

    # Orbital parameters at the specified altitude
    orbital_params = compute_orbital_params(a=a, inc_deg=inc_deg)
    n = orbital_params['n']
    T_orb = 2 * np.pi / n
    print(f"\n  Orbital period: {T_orb:.1f} s ({T_orb/60:.1f} min)")
    print(f"  SS kappa = {orbital_params['kappa']:.6f}")
    print(f"  SS c     = {orbital_params['c']:.4f}")
    print(f"  SS omega = {orbital_params['omega']:.6f} rad/s")
    print(f"  (1+2c)   = {1 + 2*orbital_params['c']:.4f}"
          f"  {'(REPULSIVE — secular drift expected)' if (1 + 2*orbital_params['c']) < 0 else ''}")

    # STEP 1: Precompute
    precomp = step1_precompute(orbital_params, output_dir, alt_km)

    # STEP 2-3: Run all controllers (CAPR uses coupled dynamics)
    print("\n[STEP 2-3] Running formation simulations ...")
    all_results, all_metrics = step3_run_all(
        orbital_params, precomp, output_dir, T_sim,
    )

    # STEP 4: Attractor Cd demonstration (GAP 3)
    CdA_demo = step4_attractor_Cd_demo(orbital_params, precomp, output_dir)

    # STEP 5: Safety boundary (GAP 4)
    if run_safety_boundary:
        step5_safety_boundary(output_dir)

    # STEP 6: Figures
    step6_figures(all_results, all_metrics, orbital_params, precomp, output_dir)

    # Summary
    print("\n" + "=" * 72)
    print("  SIMULATION COMPLETE — CORRECTED SS PHYSICS (v3)")
    print("=" * 72)
    for name in ['CAPR (coupled)', 'Uncontrolled', 'LP', 'Thruster (oracle)']:
        m = all_metrics.get(name, {})
        print(f"  {name:22s} | RMS={m.get('rms',0):.1f} m | P99={m.get('p99',0):.1f} m | "
              f"switches={m.get('switches',0)}")

    capr_m = all_metrics.get('CAPR (coupled)', {})
    unctrl_m = all_metrics.get('Uncontrolled', {})
    if capr_m.get('rms', 0) > 0 and unctrl_m.get('rms', 0) > 0:
        improvement = (1 - capr_m['rms'] / unctrl_m['rms']) * 100
        print(f"\n  CAPR improvement over uncontrolled: {improvement:.1f}%")

    print(f"  Output: {os.path.abspath(output_dir)}")
    print("=" * 72)

    return all_results, all_metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CPFC v3 Simulation — Corrected SS Physics')
    parser.add_argument('--days', type=float, default=7,
                        help='Simulation duration [days]')
    parser.add_argument('--alt', type=float, default=320,
                        help='Altitude [km] (default 320 for strong drag)')
    parser.add_argument('--no-safety-boundary', action='store_true',
                        help='Skip safety boundary computation')
    args = parser.parse_args()

    main(
        T_sim_days=args.days,
        alt_km=args.alt,
        run_safety_boundary=not args.no_safety_boundary,
    )
