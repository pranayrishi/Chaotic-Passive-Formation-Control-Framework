"""3D formation geometry visualization and animation."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
# [ADDED] Import for solar sensitivity & feasibility plots (DuBois et al. connection)
from cpfc_simulation.config import MU_EARTH, R_EARTH, J2, MASS_SAT, A_STOWED, A_DEPLOYED


def setup_publication_style():
    plt.rcParams.update({
        'font.size': 12, 'font.family': 'serif',
        'axes.labelsize': 14, 'axes.titlesize': 14,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight'
    })


def plot_formation_3d(states, labels=None, title='Formation Geometry', save_path=None):
    """
    3D plot of formation at a single time instant.
    states: (N_sats, 6) — [x, xdot, y, ydot, z, zdot]
    """
    setup_publication_style()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    for i in range(len(states)):
        x, y, z = states[i, 0], states[i, 2], states[i, 4]
        color = colors[i % len(colors)]
        label = labels[i] if labels else f'Sat {i}'
        ax.scatter(x, y, z, c=color, s=100, label=label, edgecolors='k', zorder=5)

    # Chief at origin
    ax.scatter(0, 0, 0, c='gold', s=200, marker='*', label='Chief',
               edgecolors='k', zorder=10)

    ax.set_xlabel('Radial x [m]')
    ax.set_ylabel('Along-track y [m]')
    ax.set_zlabel('Cross-track z [m]')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig, ax


def plot_formation_evolution(states_history, time_indices, time_labels,
                               target_states=None, save_path=None):
    """
    Multi-panel 3D formation at different times.
    states_history: (N_timesteps, N_sats, 6)
    time_indices: list of indices to plot
    time_labels: list of labels (e.g., ['t=0', 't=7d', 't=15d', 't=30d'])
    """
    setup_publication_style()
    n_panels = len(time_indices)
    fig = plt.figure(figsize=(5*n_panels, 5))

    for p, (tidx, label) in enumerate(zip(time_indices, time_labels)):
        ax = fig.add_subplot(1, n_panels, p+1, projection='3d')
        states = states_history[tidx]

        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
        for i in range(len(states)):
            ax.scatter(states[i, 0], states[i, 2], states[i, 4],
                       c=colors[i % len(colors)], s=60, edgecolors='k')

        if target_states is not None:
            tgt = target_states[tidx] if target_states.ndim == 3 else target_states
            for i in range(len(tgt)):
                ax.scatter(tgt[i, 0], tgt[i, 2], tgt[i, 4],
                           c='gray', s=30, alpha=0.4, marker='x')

        ax.scatter(0, 0, 0, c='gold', s=100, marker='*', edgecolors='k')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        ax.set_title(label)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_relative_motion_2d(states_history, sat_idx=0, save_path=None):
    """
    2D in-plane (x vs y) relative motion trace for one satellite.
    states_history: (N_timesteps, N_sats, 6) or (N_timesteps, 6)
    """
    setup_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if states_history.ndim == 3:
        data = states_history[:, sat_idx, :]
    else:
        data = states_history

    x = data[:, 0]
    y = data[:, 2]
    z = data[:, 4]

    # In-plane
    colors = np.arange(len(x))
    axes[0].scatter(y, x, c=colors, cmap='plasma', s=0.5, alpha=0.7)
    axes[0].set_xlabel('Along-track y [m]')
    axes[0].set_ylabel('Radial x [m]')
    axes[0].set_title('In-Plane Relative Motion')
    axes[0].set_aspect('equal')

    # Out-of-plane
    axes[1].scatter(y, z, c=colors, cmap='plasma', s=0.5, alpha=0.7)
    axes[1].set_xlabel('Along-track y [m]')
    axes[1].set_ylabel('Cross-track z [m]')
    axes[1].set_title('Out-of-Plane Relative Motion')
    axes[1].set_aspect('equal')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig, axes


def plot_formation_error_history(time_array, error_history, controller_name='CAPR',
                                   save_path=None):
    """Plot formation error vs time."""
    setup_publication_style()
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.plot(time_array / 86400, error_history, 'b-', linewidth=0.8, label=controller_name)
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Formation RMS Error [m]')
    ax.set_title('Formation Keeping Performance')
    ax.legend()
    ax.set_xlim(0, time_array[-1]/86400)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════════
# [ADDED] SOLAR ACTIVITY TRADEOFF PLOTS — DuBois et al. (2026) Connection
# ─────────────────────────────────────────────────────────────────────────────
# The following two functions connect the DORA paper's findings (solar activity
# shortens CubeSat lifetime) to CPFC's formation control framework (solar
# activity increases differential drag authority).  They visualise the tradeoff:
#   - plot_solar_sensitivity():  dual-axis lifetime vs. control authority
#   - plot_feasibility_map():    altitude × F10.7 heatmap with lifetime contour
# ═══════════════════════════════════════════════════════════════════════════════


def plot_solar_sensitivity(F107_range, lifetimes_days, formation_errors,
                           h_km=None, save_path=None):
    """Dual-axis plot: formation error vs lifetime as a function of F10.7.

    Demonstrates the core tradeoff from DuBois et al. (2026):
    - Higher F10.7 -> more drag -> shorter lifetime (paper's concern)
    - Higher F10.7 -> more differential drag -> better formation control

    Parameters
    ----------
    F107_range : ndarray
        Solar flux values [SFU].
    lifetimes_days : ndarray
        Estimated orbital lifetime for each F10.7 [days].
    formation_errors : ndarray
        Formation RMS error (or dfy_max) for each F10.7.
    h_km : float or None
        Altitude for title annotation.
    save_path : str or None
        Output file path.

    Returns
    -------
    fig, (ax1, ax2)
    """
    setup_publication_style()
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_life = '#d62728'
    color_ctrl = '#1f77b4'

    # Left axis: lifetime
    ax1.semilogy(F107_range, lifetimes_days, 's-', color=color_life,
                 linewidth=2, markersize=5, label='Orbital lifetime')
    ax1.set_xlabel(r'Solar flux $F_{10.7}$ [SFU]', fontsize=13)
    ax1.set_ylabel('Estimated orbital lifetime [days]', fontsize=12,
                   color=color_life)
    ax1.tick_params(axis='y', labelcolor=color_life)
    ax1.axhline(y=30, color=color_life, linestyle=':', alpha=0.4,
                label='30-day mission')

    # Right axis: formation control authority
    ax2 = ax1.twinx()
    ax2.plot(F107_range, formation_errors * 1e6, 'o-', color=color_ctrl,
             linewidth=2, markersize=5, label=r'$\Delta f_{y,\mathrm{max}}$')
    ax2.set_ylabel(r'Max differential drag $\Delta f_y$ [$\mu$m/s$^2$]',
                   fontsize=12, color=color_ctrl)
    ax2.tick_params(axis='y', labelcolor=color_ctrl)

    # Feasibility shading
    feasible_mask = lifetimes_days >= 30
    if np.any(feasible_mask) and np.any(~feasible_mask):
        # Find boundary F10.7
        crossings = np.where(np.diff(feasible_mask.astype(int)))[0]
        if len(crossings) > 0:
            f107_boundary = 0.5 * (F107_range[crossings[0]] +
                                    F107_range[crossings[0] + 1])
            ax1.axvspan(f107_boundary, F107_range[-1], alpha=0.08,
                        color='red', label='Lifetime < 30 days')
            ax1.axvspan(F107_range[0], f107_boundary, alpha=0.08,
                        color='green', label='Mission feasible')

    alt_str = f' at {h_km:.0f} km' if h_km is not None else ''
    ax1.set_title(f'Solar Activity Tradeoff{alt_str}\n'
                  f'(Higher F10.7: better control BUT shorter lifetime)',
                  fontsize=13)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right',
               fontsize=9)
    ax1.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig, (ax1, ax2)


def plot_feasibility_map(h_range_km, F107_range, lifetime_grid, dfy_max_grid,
                         mission_days=30.0, save_path=None):
    """2-panel heatmap: lifetime and control authority over altitude x F10.7.

    Left panel: orbital lifetime with mission-duration contour.
    Right panel: maximum differential drag authority.

    Parameters
    ----------
    h_range_km : ndarray
        Altitude range [km].
    F107_range : ndarray
        Solar flux range [SFU].
    lifetime_grid : ndarray, shape (nh, nf)
        Lifetime in days at each grid point.
    dfy_max_grid : ndarray, shape (nh, nf)
        Max differential drag [m/s^2].
    mission_days : float
        Planned mission duration for contour overlay.
    save_path : str or None
        Output file path.

    Returns
    -------
    fig, axes
    """
    setup_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # --- Left: Lifetime ---
    ax = axes[0]
    # Use log scale for lifetime since it spans orders of magnitude
    from matplotlib.colors import LogNorm
    lt_plot = np.clip(lifetime_grid, 1, None)
    im0 = ax.pcolormesh(F107_range, h_range_km, lt_plot,
                         cmap='RdYlGn', shading='auto',
                         norm=LogNorm(vmin=max(lt_plot.min(), 1),
                                      vmax=lt_plot.max()))
    plt.colorbar(im0, ax=ax, label='Orbital lifetime [days]')

    # Mission duration contour
    cs = ax.contour(F107_range, h_range_km, lifetime_grid,
                     levels=[mission_days], colors='black',
                     linewidths=2.5, linestyles='-')
    ax.clabel(cs, fmt=f'{mission_days:.0f} days', fontsize=10)

    ax.set_xlabel(r'Solar flux $F_{10.7}$ [SFU]', fontsize=12)
    ax.set_ylabel('Altitude [km]', fontsize=12)
    ax.set_title('Orbital Lifetime\n(DuBois et al. SMAD model)', fontsize=12)

    # --- Right: Control authority ---
    ax = axes[1]
    im1 = ax.pcolormesh(F107_range, h_range_km, dfy_max_grid * 1e6,
                         cmap='viridis', shading='auto')
    plt.colorbar(im1, ax=ax, label=r'$\Delta f_{y,\mathrm{max}}$ [$\mu$m/s$^2$]')

    # Overlay the lifetime contour from left panel
    ax.contour(F107_range, h_range_km, lifetime_grid,
               levels=[mission_days], colors='white',
               linewidths=2.5, linestyles='--')

    ax.set_xlabel(r'Solar flux $F_{10.7}$ [SFU]', fontsize=12)
    ax.set_title('Differential Drag Authority\n'
                 '(white dashed = lifetime boundary)', fontsize=12)

    fig.suptitle('Mission Feasibility: Lifetime vs. Control Authority',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig, axes
