"""Monte Carlo uncertainty analysis for CPFC robustness assessment."""
import numpy as np
from multiprocessing import Pool
from cpfc_simulation.config import F107_NOMINAL, AP_NOMINAL


def sample_atmospheric_params(n_samples, F107_mean=F107_NOMINAL, F107_std=22.5,
                                Ap_mean=AP_NOMINAL, Ap_std=5.0, seed=42):
    """
    Sample atmospheric parameters from Gaussian distributions.
    F107 uncertainty: ±15% → std = 0.15 * 150 = 22.5
    Ap uncertainty: ±50% → std = 0.5 * 10 = 5.0
    """
    rng = np.random.default_rng(seed)
    F107_samples = np.clip(rng.normal(F107_mean, F107_std, n_samples), 70, 400)
    Ap_samples = np.clip(rng.normal(Ap_mean, Ap_std, n_samples), 0, 400).astype(int)
    return F107_samples, Ap_samples


def run_single_mc(params, simulation_func):
    """
    Run a single Monte Carlo realization.
    params: dict with F107, Ap, and any other varied parameters
    simulation_func: callable that takes params and returns metrics dict
    """
    try:
        metrics = simulation_func(params)
        return {'success': True, 'metrics': metrics, 'params': params}
    except Exception as e:
        return {'success': False, 'error': str(e), 'params': params}


def monte_carlo_analysis(simulation_func, n_samples=100, n_workers=4, seed=42,
                          extra_params=None):
    """
    Run Monte Carlo analysis over atmospheric uncertainty.

    simulation_func: callable(params_dict) -> metrics_dict
    Returns: dict with statistics over all MC runs
    """
    F107_samples, Ap_samples = sample_atmospheric_params(n_samples, seed=seed)

    params_list = []
    for i in range(n_samples):
        p = {'F107': float(F107_samples[i]), 'Ap': int(Ap_samples[i]), 'mc_index': i}
        if extra_params:
            p.update(extra_params)
        params_list.append(p)

    # Run simulations (serial for safety — simulation_func may not be picklable)
    results = []
    for p in params_list:
        res = run_single_mc(p, simulation_func)
        results.append(res)

    # Aggregate statistics
    successful = [r for r in results if r['success']]
    n_success = len(successful)

    if n_success == 0:
        return {'n_total': n_samples, 'n_success': 0, 'error': 'All runs failed'}

    # Extract metric arrays
    metric_keys = list(successful[0]['metrics'].keys())
    stats = {'n_total': n_samples, 'n_success': n_success}

    for key in metric_keys:
        vals = []
        for s in successful:
            v = s['metrics'].get(key)
            if isinstance(v, (int, float, np.floating, np.integer)):
                vals.append(float(v))

        if vals:
            vals = np.array(vals)
            stats[key] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'median': float(np.median(vals)),
                'p5': float(np.percentile(vals, 5)),
                'p95': float(np.percentile(vals, 95)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals))
            }

    # Check chaos condition satisfaction
    chaos_satisfied = sum(1 for s in successful
                          if s['metrics'].get('chaos_exists', False))
    stats['chaos_satisfaction_pct'] = 100.0 * chaos_satisfied / n_success if n_success > 0 else 0.0

    return stats


def mc_comparison(capr_stats, lp_stats, metric_name='rms'):
    """Compare two controllers' MC statistics."""
    if metric_name in capr_stats and metric_name in lp_stats:
        capr_m = capr_stats[metric_name]
        lp_m = lp_stats[metric_name]
        improvement = (lp_m['mean'] - capr_m['mean']) / lp_m['mean'] * 100 if lp_m['mean'] != 0 else 0
        return {
            'capr_mean': capr_m['mean'], 'capr_p95': capr_m['p95'],
            'lp_mean': lp_m['mean'], 'lp_p95': lp_m['p95'],
            'improvement_pct': improvement
        }
    return None
