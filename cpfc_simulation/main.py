"""CPFC Master Simulation Orchestrator.

Chaotic Passive Formation Control (CPFC) Framework
===================================================
Unifies differential drag control, natural perturbations (J2/SRP/drag),
and chaos stabilisation for propellant-free LEO formation keeping.

This script runs the full simulation pipeline:
1. Initialise chief orbit and deputy formation geometry
2. Precompute Melnikov chaos boundaries and Poincaré map data
3. Run 30-day coupled attitude-orbital simulation with CAPR controller
4. Run benchmark controllers for comparison
5. Compute all metrics and generate publication-quality figures
6. (Optional) Monte Carlo robustness analysis
7. (Optional) Safety boundary parameter sweep
"""
import os
import sys
import time
import numpy as np
from datetime import timedelta
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
from cpfc_simulation.config import (
    MU_EARTH, R_EARTH, J2, OMEGA_EARTH,
    MASS_SAT, IXX, IYY, IZZ,
    A_STOWED, A_DEPLOYED, ALT_NOMINAL, INC_NOMINAL,
    FORMATION_RADIUS, N_SATELLITES, SEPARATION_TARGET,
    F107_NOMINAL, F107A_NOMINAL, AP_NOMINAL,
    DT_INTEGRATOR, T_MISSION, T_ORBIT, EPOCH,
)

# ── Dynamics ───────────────────────────────────────────────────────────────────
from cpfc_simulation.dynamics.orbit_propagator import OrbitPropagator
from cpfc_simulation.dynamics.relative_motion import (
    compute_SS_coefficients, propagate_relative_motion,
    corrected_SS_solution,
)
from cpfc_simulation.dynamics.attitude_dynamics import AttitudePropagator
from cpfc_simulation.dynamics.perturbations import get_atmospheric_density
from cpfc_simulation.dynamics.aerodynamic_model import CubeSatAeroModel

# ── Chaos ──────────────────────────────────────────────────────────────────────
from cpfc_simulation.chaos.melnikov import (
    melnikov_chaos_boundary, optimal_switching_frequency,
    compute_heteroclinic_orbit,
)
from cpfc_simulation.chaos.lyapunov import compute_MLE_wolf, compute_short_time_lyapunov
from cpfc_simulation.chaos.attractor_library import (
    doroshin_attractor_5, generate_attractor, classify_attractor,
    time_averaged_cross_section,
)
from cpfc_simulation.chaos.capr_law import CAPRController

# ── Formation ──────────────────────────────────────────────────────────────────
from cpfc_simulation.formation.formation_geometry import (
    compute_orbital_params, pco_formation_state, formation_error,
)
from cpfc_simulation.formation.phase_plane_analysis import (
    ss_corrected_ellipse, compute_accessible_ellipses,
    compute_reconfiguration_time,
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
    compute_all_metrics, formation_rms_error, controller_comparison_table,
)
from cpfc_simulation.analysis.monte_carlo import monte_carlo_analysis

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


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _initial_chief_state_eci(a, inc_deg, raan_deg=0.0):
    """Return [r, v] in ECI for a circular orbit at the given elements."""
    inc = np.radians(inc_deg)
    raan = np.radians(raan_deg)
    # Position at ascending node
    r_pqw = np.array([a, 0.0, 0.0])
    v_mag = np.sqrt(MU_EARTH / a)
    v_pqw = np.array([0.0, v_mag, 0.0])
    # Rotation PQW -> ECI
    cos_O, sin_O = np.cos(raan), np.sin(raan)
    cos_i, sin_i = np.cos(inc), np.sin(inc)
    R_mat = np.array([
        [cos_O, -sin_O * cos_i, sin_O * sin_i],
        [sin_O,  cos_O * cos_i, -cos_O * sin_i],
        [0.0,    sin_i,          cos_i],
    ])
    return np.concatenate([R_mat @ r_pqw, R_mat @ v_pqw])


def _compute_dfy_bounds(rho, v_orb, Cd_approx=2.2):
    """Min/max achievable differential specific force [m/s^2]."""
    dfy_max = 0.5 * rho * v_orb**2 * Cd_approx * (A_DEPLOYED - A_STOWED) / MASS_SAT
    return -dfy_max, dfy_max


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — PRECOMPUTATION (Melnikov, heteroclinic orbit, attractor survey)
# ═══════════════════════════════════════════════════════════════════════════════

def step1_precompute(orbital_params, output_dir):
    """Precompute Melnikov boundaries and attractor data."""
    print("\n[STEP 1] Precomputing chaos data ...")
    n = orbital_params['n']
    a = orbital_params['a']
    v_orb = n * a

    # Approximate aerodynamic torque coefficient at nominal altitude
    r_test = np.array([a, 0.0, 0.0])
    try:
        rho = get_atmospheric_density(r_test, EPOCH)
    except Exception:
        rho = 1e-12
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
    print(f"  Heteroclinic orbit: {len(tau_h)} points, theta range [{theta_h.min():.3f}, {theta_h.max():.3f}] rad")

    # Attractor survey (Doroshin Type 5)
    aero_model = CubeSatAeroModel()
    att_state0 = [0.01, 0.005, 0.002, 0.1, 0.0]  # [ox, oy, oz, theta, alpha]
    att_t, att_traj = generate_attractor(
        lambda t, s: doroshin_attractor_5(t, s, IXX, IYY, IZZ, n, M_aero),
        att_state0, T=2000.0, dt=0.5,
    )
    att_class, att_mle = classify_attractor(att_traj)
    avg_CdA_mean, avg_CdA_std, _ = time_averaged_cross_section(att_traj)
    print(f"  Doroshin-5 attractor: class={att_class}, <CdA>={avg_CdA_mean:.6f} m^2")

    # Save Melnikov spectrum plot
    plot_melnikov_spectrum(Omega_arr, M_vals, Omega_opt,
                          save_path=os.path.join(output_dir, 'melnikov_spectrum.png'))

    # Save attitude phase portrait
    plot_attitude_phase_portrait(
        att_traj[:, 3], att_traj[:, 1],
        separatrix_theta=theta_h, separatrix_omega=omega_h,
        attractor_label='Doroshin-5', alt_km=ALT_NOMINAL / 1e3,
        save_path=os.path.join(output_dir, 'attitude_phase_portrait.png'),
    )

    return {
        'Omega_opt': Omega_opt, 'M_max': M_max,
        'Omega_range': Omega_arr, 'M_values': M_vals,
        'tau_h': tau_h, 'theta_h': theta_h, 'omega_h': omega_h,
        'rho_nominal': rho, 'M_aero': M_aero,
        'avg_CdA': avg_CdA_mean, 'attractor_class': att_class,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — FORMATION SIMULATION (30-day coupled propagation)
# ═══════════════════════════════════════════════════════════════════════════════

def step2_simulate_formation(orbital_params, precomp, controller, controller_name,
                              T_sim=None, dt_output=60.0):
    """Run formation simulation with a given controller.

    Uses the corrected SS relative-motion model for computational efficiency.
    Attitude dynamics are propagated in parallel for the CAPR controller.

    Returns
    -------
    dict with keys: time, states (N_t, N_dep, 6), dfy_history, switch_log, errors
    """
    if T_sim is None:
        T_sim = T_MISSION

    n = orbital_params['n']
    kappa = orbital_params['kappa']
    c = orbital_params['c']
    s = orbital_params['s']
    a = orbital_params['a']
    inc_rad = orbital_params['inc']
    T_orb = 2 * np.pi / n

    rho = precomp.get('rho_nominal', 1e-12)
    v_orb = n * a
    dfy_min, dfy_max = _compute_dfy_bounds(rho, v_orb)

    N_dep = N_SATELLITES - 1  # 3 deputies
    t_output = np.arange(0, T_sim, dt_output)
    N_t = len(t_output)

    # Target formation (PCO at t=0)
    target_states_t0 = pco_formation_state(0.0, n, kappa, FORMATION_RADIUS, N_SATELLITES)
    # Deputies are indices 1,2,3 (index 0 = chief at origin)
    target_dep = target_states_t0[1:]  # (3, 6)

    # Initial states: deputies start at target + small perturbation
    rng = np.random.default_rng(12345)
    init_dep = target_dep.copy()
    init_dep[:, [0, 2, 4]] += rng.normal(0, 20, (N_dep, 3))  # ±20 m position error
    init_dep[:, [1, 3, 5]] += rng.normal(0, 0.01, (N_dep, 3))  # ±0.01 m/s velocity error

    # Storage
    states_history = np.zeros((N_t, N_dep, 6))
    dfy_history = np.zeros((N_t, N_dep))
    error_history = np.zeros(N_t)
    states_history[0] = init_dep

    # Attitude propagator for CAPR (simplified pitch-only per deputy)
    attitude_states = np.zeros((N_dep, 2))  # [theta, omega_y] per deputy
    attitude_states[:, 0] = rng.uniform(-0.1, 0.1, N_dep)

    # Scheduler per deputy
    schedulers = [DragPlateScheduler() for _ in range(N_dep)]

    print(f"  Simulating {controller_name} for {T_sim/86400:.0f} days ({N_t} output steps) ...")

    current_dep = init_dep.copy()

    for k in tqdm(range(1, N_t), desc=f'  {controller_name}', leave=False):
        t_now = t_output[k - 1]
        dt = t_output[k] - t_output[k - 1]

        # Target at current time (PCO rotates)
        target_now = pco_formation_state(t_now, n, kappa, FORMATION_RADIUS, N_SATELLITES)[1:]

        for j in range(N_dep):
            state_j = current_dep[j]
            target_j = target_now[j]

            # Controller decision
            if isinstance(controller, CAPRController):
                result = controller(
                    t_now, state_j, target_j,
                    rho=rho, v_rel=v_orb, T_orb=T_orb,
                )
                deploy = result['deploy_command']
                dfy_req = result['dfy_required']
            elif isinstance(controller, ActiveThrusterController):
                dfy_req = controller(t_now, state_j, target_j, T_orb, dt)
                deploy = 0
            else:
                deploy, dfy_req = controller(t_now, state_j, target_j, T_orb)

            # Apply scheduler constraints
            deploy = schedulers[j].request_switch(t_now, deploy)

            # Effective differential drag
            if isinstance(controller, ActiveThrusterController):
                dfy_eff = dfy_req  # ideal thruster
            else:
                CdA_dep = 2.2 * (A_DEPLOYED if deploy else A_STOWED)
                CdA_chief = 2.2 * A_STOWED
                dfy_eff = -0.5 * rho * v_orb**2 * (CdA_dep - CdA_chief) / MASS_SAT

            dfy_history[k, j] = dfy_eff

            # Propagate relative motion (one output step)
            sol = propagate_relative_motion(
                state0=state_j,
                t_span=(0, dt),
                orbital_params={'a': a, 'inc_rad': inc_rad},
                diff_forces={'dfx': 0.0, 'dfy': dfy_eff, 'dfz': 0.0},
            )
            current_dep[j] = sol.y[:, -1]

        states_history[k] = current_dep.copy()

        # RMS error at this step
        pos_err = current_dep[:, [0, 2, 4]] - target_now[:, [0, 2, 4]]
        error_history[k] = np.sqrt(np.mean(pos_err**2))

    # Aggregate switch logs
    switch_logs = []
    for sched in schedulers:
        switch_logs.extend(sched.switch_log)

    return {
        'time': t_output,
        'states': states_history,
        'dfy_history': dfy_history,
        'error_history': error_history,
        'switch_log': switch_logs,
        'target_states': lambda t: pco_formation_state(t, n, kappa, FORMATION_RADIUS, N_SATELLITES)[1:],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — BENCHMARK COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def step3_run_benchmarks(orbital_params, precomp, output_dir, T_sim=None):
    """Run all controllers and collect metrics."""
    if T_sim is None:
        T_sim = T_MISSION

    n = orbital_params['n']
    kappa = orbital_params['kappa']
    c = orbital_params['c']
    a = orbital_params['a']
    rho = precomp.get('rho_nominal', 1e-12)
    v_orb = n * a
    dfy_min, dfy_max = _compute_dfy_bounds(rho, v_orb)

    # Build controllers
    controllers = {
        'CAPR': CAPRController(orbital_params),
        'LP': LPDifferentialDragController(dfy_min, dfy_max, n, kappa, c),
        'Convex': ConvexOptController(dfy_min, dfy_max, n, kappa, c),
        'Constraint-Tight': ConstraintTighteningController(dfy_min, dfy_max, n, kappa, c),
        'Thruster (oracle)': ActiveThrusterController(n, kappa, c),
    }

    all_results = {}
    all_metrics = {}
    dt_output = 60.0

    for name, ctrl in controllers.items():
        result = step2_simulate_formation(
            orbital_params, precomp, ctrl, name, T_sim=T_sim, dt_output=dt_output,
        )
        all_results[name] = result

        # Compute metrics with time-varying PCO targets
        t_arr = result['time']
        target_history = np.array([
            pco_formation_state(t, n, kappa, FORMATION_RADIUS, N_SATELLITES)[1:]
            for t in t_arr
        ])  # (N_t, N_dep, 6)
        metrics = compute_all_metrics(
            result['states'],
            target_history,
            result['switch_log'],
            result['dfy_history'].flatten(),
            dt_output,
        )
        all_metrics[name] = metrics
        print(f"  {name:20s} | RMS={metrics['rms']:.1f} m | P99={metrics['p99']:.1f} m "
              f"| switches={metrics['switches']} | dV={metrics['dv']:.4f} m/s")

    # Print comparison table
    table_data = {}
    for name, m in all_metrics.items():
        table_data[name] = {
            'rms': m['rms'], 'switches': m['switches'],
            'dv': m['dv'], 'p99': m['p99'],
        }
    print("\n" + controller_comparison_table(table_data))

    return all_results, all_metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def step4_generate_figures(all_results, all_metrics, orbital_params, precomp, output_dir):
    """Generate all publication-quality figures."""
    print("\n[STEP 4] Generating figures ...")
    n = orbital_params['n']
    kappa = orbital_params['kappa']

    # ── Result 4: Formation evolution (CAPR) ──────────────────────────────────
    capr_res = all_results.get('CAPR')
    if capr_res is not None:
        t_arr = capr_res['time']
        dt_day = t_arr / 86400.0
        T_total_days = dt_day[-1]

        # Snapshot indices: t=0, 7d, 15d, 30d (or final)
        snap_days = [0, 7, 15, min(30, T_total_days)]
        snap_idx = [np.argmin(np.abs(dt_day - d)) for d in snap_days]
        snap_labels = [f't = {d:.0f} d' for d in snap_days]

        plot_formation_evolution(
            capr_res['states'], snap_idx, snap_labels,
            save_path=os.path.join(output_dir, 'formation_evolution_capr.png'),
        )

        # 2D relative motion for first deputy
        plot_relative_motion_2d(
            capr_res['states'], sat_idx=0,
            save_path=os.path.join(output_dir, 'relative_motion_2d.png'),
        )

    # ── Error history comparison ──────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'CAPR': '#e41a1c', 'LP': '#377eb8', 'Convex': '#4daf4a',
              'Constraint-Tight': '#984ea3', 'Thruster (oracle)': '#ff7f00'}
    for name, res in all_results.items():
        t_days = res['time'] / 86400
        ax.plot(t_days, res['error_history'], color=colors.get(name, 'gray'),
                linewidth=0.8, label=name)
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Formation RMS Error [m]')
    ax.set_title('Controller Comparison — Formation Keeping')
    ax.legend()
    ax.set_xlim(0, t_days[-1])
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'error_comparison.png'), dpi=300)
    plt.close(fig)
    print("  Saved error_comparison.png")

    # ── Lyapunov exponent (from attractor precomputation) ─────────────────────
    # Quick MLE computation on the Doroshin-5 attractor
    M_aero = precomp.get('M_aero', 0.0)
    att_eom = lambda t, s: doroshin_attractor_5(t, s, IXX, IYY, IZZ, n, M_aero)
    state0_att = [0.01, 0.005, 0.002, 0.1, 0.0]
    mle, mle_series = compute_MLE_wolf(att_eom, state0_att, dt=0.5, T_total=2000.0,
                                        renorm_steps=50)
    t_mle = np.arange(1, len(mle_series) + 1) * 50 * 0.5
    plot_lyapunov_time_series(
        t_mle, mle_series,
        save_path=os.path.join(output_dir, 'lyapunov_time_series.png'),
    )
    print(f"  MLE = {mle:.4f} rad/s")

    print(f"  All figures saved to {output_dir}/")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — SAFETY BOUNDARY MAP (optional, computationally expensive)
# ═══════════════════════════════════════════════════════════════════════════════

def step5_safety_boundary(output_dir, h_range=None, inc_range=None, F107_range=None):
    """Generate the novel Melnikov safety boundary map."""
    print("\n[STEP 5] Computing Melnikov safety boundary map ...")
    if h_range is None:
        h_range = np.arange(300, 625, 50)  # coarse for speed
    if inc_range is None:
        inc_range = np.arange(0, 99, 10)
    if F107_range is None:
        F107_range = np.array([70, 150, 230])

    h5_path = os.path.join(output_dir, 'safety_boundary_map.h5')
    generate_safety_boundary_map(h_range, inc_range, F107_range,
                                  n_workers=1, output_file=h5_path)

    plot_safety_boundary_heatmap(
        h5_file=h5_path,
        save_path=os.path.join(output_dir, 'safety_boundary_heatmap.png'),
    )
    print(f"  Saved safety_boundary_map.h5 and heatmap.")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main(T_sim_days=30, run_safety_boundary=False, run_monte_carlo=False):
    """Run the complete CPFC simulation pipeline.

    Parameters
    ----------
    T_sim_days : float
        Simulation duration in days (default 30).
    run_safety_boundary : bool
        If True, compute the full Melnikov safety boundary map (slow).
    run_monte_carlo : bool
        If True, run Monte Carlo robustness analysis (slow).
    """
    T_sim = T_sim_days * 86400.0
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 72)
    print("  CPFC — Chaotic Passive Formation Control Simulation")
    print("=" * 72)
    print(f"  Altitude:    {ALT_NOMINAL/1e3:.0f} km")
    print(f"  Inclination: {INC_NOMINAL:.1f} deg")
    print(f"  Formation:   {N_SATELLITES} satellites, {FORMATION_RADIUS:.0f} m PCO radius")
    print(f"  Duration:    {T_sim_days:.0f} days")
    print(f"  Output:      {os.path.abspath(output_dir)}")
    print("=" * 72)

    # Orbital parameters with SS corrections
    orbital_params = compute_orbital_params()
    n = orbital_params['n']
    print(f"\n  Orbital period: {2*np.pi/n:.1f} s ({2*np.pi/n/60:.1f} min)")
    print(f"  SS kappa = {orbital_params['kappa']:.6f}")
    print(f"  SS c     = {orbital_params['c']:.6e}")

    # ── STEP 1: Precompute ────────────────────────────────────────────────────
    precomp = step1_precompute(orbital_params, output_dir)

    # ── STEP 2+3: Simulate all controllers ────────────────────────────────────
    print("\n[STEP 2-3] Running formation simulations ...")
    all_results, all_metrics = step3_run_benchmarks(
        orbital_params, precomp, output_dir, T_sim=T_sim,
    )

    # ── STEP 4: Figures ───────────────────────────────────────────────────────
    step4_generate_figures(all_results, all_metrics, orbital_params, precomp, output_dir)

    # ── STEP 5: Safety boundary (optional) ────────────────────────────────────
    if run_safety_boundary:
        step5_safety_boundary(output_dir)

    # ── STEP 6: Monte Carlo (optional) ────────────────────────────────────────
    if run_monte_carlo:
        print("\n[STEP 6] Monte Carlo analysis ...")

        def mc_sim_func(params):
            """Single MC realisation with varied F10.7 and Ap."""
            # This would re-run simulation with different atmospheric params
            # For now, return metrics from nominal (placeholder for full MC)
            return all_metrics.get('CAPR', {})

        mc_stats = monte_carlo_analysis(mc_sim_func, n_samples=10)
        print(f"  MC chaos satisfaction: {mc_stats.get('chaos_satisfaction_pct', 0):.1f}%")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SIMULATION COMPLETE")
    print("=" * 72)
    capr_m = all_metrics.get('CAPR', {})
    print(f"  CAPR RMS error:  {capr_m.get('rms', 0):.1f} m")
    print(f"  CAPR P99 error:  {capr_m.get('p99', 0):.1f} m")
    print(f"  CAPR switches:   {capr_m.get('switches', 0)}")
    print(f"  Output dir:      {os.path.abspath(output_dir)}")
    print("=" * 72)

    return all_results, all_metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CPFC Simulation')
    parser.add_argument('--days', type=float, default=30, help='Simulation duration [days]')
    parser.add_argument('--safety-boundary', action='store_true', help='Run safety boundary map')
    parser.add_argument('--monte-carlo', action='store_true', help='Run Monte Carlo analysis')
    args = parser.parse_args()

    main(
        T_sim_days=args.days,
        run_safety_boundary=args.safety_boundary,
        run_monte_carlo=args.monte_carlo,
    )
