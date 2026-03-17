"""Formation keeping metrics and performance analysis."""
import numpy as np


def formation_rms_error(positions_rel, targets_rel):
    """
    Compute RMS formation keeping error.
    positions_rel: (N_timesteps, N_sats, 6) — [x,xdot,y,ydot,z,zdot]
    targets_rel: (N_timesteps, N_sats, 6) or (N_sats, 6) if constant target
    """
    if targets_rel.ndim == 2:
        targets_rel = np.broadcast_to(targets_rel, positions_rel.shape)

    pos_err = positions_rel[:, :, [0,1,2]] - targets_rel[:, :, [0,1,2]]
    rms_per_step = np.sqrt(np.mean(pos_err**2, axis=(1,2)))
    rms_total = np.sqrt(np.mean(pos_err**2))
    return {
        'rms_total': rms_total,
        'rms_time_series': rms_per_step,
        'max_error': np.max(np.abs(pos_err)),
        'p99_error': np.percentile(np.sqrt(np.sum(pos_err**2, axis=2)).flatten(), 99),
        'mean_error': np.mean(np.sqrt(np.sum(pos_err**2, axis=2)))
    }


def delta_v_equivalent(dfy_history, dt):
    """
    Compute delta-V equivalent of differential drag maneuvers.
    dfy_history: array of differential specific force values [m/s^2]
    dt: timestep [s]
    """
    return np.sum(np.abs(dfy_history)) * dt


def switch_count(switch_log):
    """Count total drag plate switches from switch log."""
    return len(switch_log)


def formation_lifetime(positions_rel, targets_rel, threshold_m=1000.0):
    """
    Compute formation lifetime: time until any satellite exceeds threshold from target.
    """
    if targets_rel.ndim == 2:
        targets_rel = np.broadcast_to(targets_rel, positions_rel.shape)

    pos_err = positions_rel[:, :, [0,1,2]] - targets_rel[:, :, [0,1,2]]
    dist_err = np.sqrt(np.sum(pos_err**2, axis=2))  # (N_timesteps, N_sats)
    max_err_per_step = np.max(dist_err, axis=1)

    exceeded = np.where(max_err_per_step > threshold_m)[0]
    if len(exceeded) > 0:
        return exceeded[0]  # index of first exceedance
    return len(max_err_per_step)  # never exceeded


def controller_comparison_table(results_dict):
    """
    Generate comparison table data for all controllers.
    results_dict: {'controller_name': {'rms': X, 'switches': X, 'dv': X, 'p99': X}}
    Returns: formatted string (LaTeX-ready)
    """
    header = "| Controller | RMS Error [m] | # Switches | ΔV-equiv [m/s] | P99 Error [m] |"
    separator = "|-----------|---------------|------------|-----------------|----------------|"
    rows = [header, separator]

    for name, data in results_dict.items():
        row = f"| {name} | {data.get('rms', 0):.1f} | {data.get('switches', 'N/A')} | {data.get('dv', 0):.4f} | {data.get('p99', 0):.1f} |"
        rows.append(row)

    return '\n'.join(rows)


def compute_all_metrics(positions_rel, targets_rel, switch_log, dfy_history, dt, threshold_m=1000.0):
    """Compute all metrics in one call."""
    rms = formation_rms_error(positions_rel, targets_rel)
    dv = delta_v_equivalent(dfy_history, dt)
    n_switches = switch_count(switch_log)
    lifetime_idx = formation_lifetime(positions_rel, targets_rel, threshold_m)

    return {
        'rms': rms['rms_total'],
        'rms_time_series': rms['rms_time_series'],
        'max_error': rms['max_error'],
        'p99': rms['p99_error'],
        'mean_error': rms['mean_error'],
        'dv': dv,
        'switches': n_switches,
        'lifetime_steps': lifetime_idx,
        'lifetime_seconds': lifetime_idx * dt
    }
