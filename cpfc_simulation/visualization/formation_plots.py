"""3D formation geometry visualization and animation."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize


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
