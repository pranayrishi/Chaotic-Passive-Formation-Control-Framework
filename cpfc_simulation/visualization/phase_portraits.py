"""Phase portrait and Poincaré section visualization."""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def plot_attitude_phase_portrait(theta_traj, omega_traj, separatrix_theta=None,
                                   separatrix_omega=None, poincare_theta=None,
                                   poincare_omega=None, attractor_label='',
                                   alt_km=450, save_path=None):
    """
    Publication-quality attitude phase portrait.
    theta_traj, omega_traj: trajectory arrays
    separatrix: dashed curve overlay
    poincare points: dots at section crossings
    """
    setup_publication_style()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Color trajectory by time
    N = len(theta_traj)
    colors = plt.cm.plasma(np.linspace(0, 1, N))

    for i in range(N-1):
        ax.plot(theta_traj[i:i+2], omega_traj[i:i+2], color=colors[i],
                linewidth=0.3, alpha=0.7)

    # Separatrix
    if separatrix_theta is not None and separatrix_omega is not None:
        ax.plot(separatrix_theta, separatrix_omega, 'w--', linewidth=1.5,
                label='Separatrix', alpha=0.8)
        ax.plot(separatrix_theta, -separatrix_omega, 'w--', linewidth=1.5, alpha=0.8)

    # Poincaré crossings
    if poincare_theta is not None and poincare_omega is not None:
        ax.scatter(poincare_theta, poincare_omega, c='lime', s=8, zorder=5,
                   label=r'Poincar\'{e} crossings', edgecolors='none')

    ax.set_xlabel(r'Pitch angle $\theta$ [rad]')
    ax.set_ylabel(r'Pitch rate $\dot{\theta}$ [rad/s]')
    ax.set_title(f'Attitude Phase Portrait — {attractor_label} ({alt_km} km)')
    ax.legend(loc='upper right')

    # Colorbar for time
    sm = ScalarMappable(cmap='plasma', norm=Normalize(0, N))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Time step')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig, ax


def plot_poincare_section(y_crossings, ydot_crossings, target_fp=None,
                           manifold_stable=None, manifold_unstable=None,
                           controlled_y=None, controlled_ydot=None,
                           save_path=None):
    """
    Poincaré section in (y, ydot) formation phase space.
    """
    setup_publication_style()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Uncontrolled crossings (chaotic sea)
    ax.scatter(y_crossings, ydot_crossings, c='steelblue', s=3, alpha=0.5,
               label='Uncontrolled')

    # Target fixed point
    if target_fp is not None:
        ax.plot(target_fp[0], target_fp[1], 'r*', markersize=15,
                label='Target fixed point', zorder=10)

    # Invariant manifolds
    if manifold_stable is not None:
        ax.plot(manifold_stable[:, 0], manifold_stable[:, 1], 'g-',
                linewidth=1.5, label='Stable manifold')
    if manifold_unstable is not None:
        ax.plot(manifold_unstable[:, 0], manifold_unstable[:, 1], 'r-',
                linewidth=1.5, label='Unstable manifold')

    # Controlled trajectory crossings
    if controlled_y is not None:
        ax.scatter(controlled_y, controlled_ydot, c='gold', s=8, zorder=5,
                   label='CAPR controlled', edgecolors='k', linewidths=0.3)

    ax.set_xlabel(r'Along-track $y$ [m]')
    ax.set_ylabel(r'Along-track rate $\dot{y}$ [m/s]')
    ax.set_title(r'Poincar\'{e} Section — Formation Phase Space')
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig, ax


def plot_lyapunov_time_series(time_array, mle_series, switch_times=None,
                                n_deputies=3, save_path=None):
    """Plot MLE vs time for all deputy satellites."""
    setup_publication_style()
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    colors = ['#e41a1c', '#377eb8', '#4daf4a']

    if isinstance(mle_series, list):
        for i, (mle, color) in enumerate(zip(mle_series, colors)):
            ax.plot(time_array[:len(mle)] / 3600, mle, color=color,
                    linewidth=1.0, label=f'Deputy {i+1}')
    else:
        ax.plot(time_array[:len(mle_series)] / 3600, mle_series, 'b-', linewidth=1.0)

    # Zero line
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    # Switch events
    if switch_times is not None:
        for st in switch_times:
            ax.axvline(x=st/3600, color='orange', alpha=0.2, linewidth=0.5)

    ax.set_xlabel('Time [hours]')
    ax.set_ylabel(r'Max Lyapunov Exponent $\lambda_{max}$ [1/s]')
    ax.set_title('Lyapunov Exponent Time Series')
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig, ax
