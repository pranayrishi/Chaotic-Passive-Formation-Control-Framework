"""Melnikov safety boundary heatmap visualization."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
# [ADDED] LogNorm used for lifetime overlay heatmap (DuBois et al. connection)
from matplotlib.colors import LogNorm


def setup_publication_style():
    plt.rcParams.update({
        'font.size': 12, 'font.family': 'serif',
        'axes.labelsize': 14, 'axes.titlesize': 14,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight'
    })


def plot_safety_boundary_heatmap(h5_file=None, Delta_y_max=None, h_range=None,
                                   inc_range=None, F107_values=None,
                                   chaos_exists=None, save_path=None):
    """
    3-panel Melnikov safety boundary map.
    One panel per solar activity level.
    x-axis: inclination, y-axis: altitude, color: max controllable separation.
    """
    setup_publication_style()

    if h5_file is not None:
        with h5py.File(h5_file, 'r') as f:
            h_range = f['h_range_km'][:]
            inc_range = f['inc_range_deg'][:]
            F107_range = f['F107_range'][:]
            Delta_y_max = f['Delta_y_max'][:]
            chaos_exists = f['chaos_exists'][:]

        # Select 3 F10.7 values: low, medium, high
        f107_indices = [0, len(F107_range)//2, len(F107_range)-1]
        F107_values = F107_range[f107_indices]
    else:
        f107_indices = list(range(min(3, Delta_y_max.shape[2])))
        if F107_values is None:
            F107_values = [70, 150, 230]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for panel, (f107_idx, f107_val) in enumerate(zip(f107_indices, F107_values)):
        ax = axes[panel]

        data = Delta_y_max[:, :, f107_idx]

        # Heatmap
        im = ax.pcolormesh(inc_range, h_range, data, cmap='viridis', shading='auto')
        plt.colorbar(im, ax=ax, label=r'$\Delta y_{max}$ [m]')

        # Chaos boundary contour
        if chaos_exists is not None:
            chaos_data = chaos_exists[:, :, f107_idx].astype(float)
            ax.contour(inc_range, h_range, chaos_data, levels=[0.5],
                       colors='white', linewidths=2, linestyles='--')

        ax.set_xlabel('Inclination [deg]')
        if panel == 0:
            ax.set_ylabel('Altitude [km]')
        ax.set_title(f'F10.7 = {f107_val:.0f} sfu')

    fig.suptitle('Melnikov Formation Safety Boundary Map', fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig, axes


def plot_chaos_region(h_range, inc_range, chaos_exists_2d, title='', save_path=None):
    """Plot chaos existence region for a single F10.7 value."""
    setup_publication_style()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    im = ax.pcolormesh(inc_range, h_range, chaos_exists_2d.astype(float),
                        cmap='RdYlGn', shading='auto', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Chaos Exists (1=Yes, 0=No)')

    ax.set_xlabel('Inclination [deg]')
    ax.set_ylabel('Altitude [km]')
    ax.set_title(f'Chaos Region — {title}')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig, ax


def plot_melnikov_spectrum(Omega_range, M_values, Omega_opt=None, save_path=None):
    """Plot Melnikov integral vs perturbation frequency."""
    setup_publication_style()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(Omega_range, M_values, 'b-', linewidth=1.5)
    ax.fill_between(Omega_range, 0, M_values, alpha=0.2, color='blue')

    if Omega_opt is not None:
        ax.axvline(x=Omega_opt, color='red', linestyle='--', linewidth=1.5,
                    label=f'Optimal: {Omega_opt:.4f} rad/s')

    ax.set_xlabel(r'Switching frequency $\Omega$ [rad/s]')
    ax.set_ylabel(r'$|M(\Omega)|$')
    ax.set_title('Melnikov Integral Spectrum')
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════════
# [ADDED] LIFETIME OVERLAY ON SAFETY BOUNDARY — DuBois et al. (2026) Connection
# ─────────────────────────────────────────────────────────────────────────────
# Overlays orbital lifetime contours (from the SMAD analytical decay model)
# on top of the Melnikov safety boundary heatmap.  This answers the question:
# "We have formation control authority at this (h, i, F10.7) point — but how
#  many days do we have before the orbit decays to re-entry?"
#
# Red contours  = estimated lifetime (30d, 60d, 90d, 180d, 365d)
# Yellow contour = mission feasibility limit (default 30 days)
# ═══════════════════════════════════════════════════════════════════════════════


def plot_safety_boundary_with_lifetime(h5_file, mission_days=30.0,
                                        Cd=2.2, mass=None, save_path=None):
    """Safety boundary heatmap with orbital lifetime contours overlaid.

    Combines the Melnikov controllability map with DuBois et al.-style
    lifetime estimates to show operators: "you have formation control
    authority here, but only X days before re-entry."

    Parameters
    ----------
    h5_file : str
        Path to safety_boundary_map.h5.
    mission_days : float
        Mission duration for feasibility contour [days].
    Cd : float
        Drag coefficient for lifetime estimation.
    mass : float or None
        Satellite mass [kg].
    save_path : str or None
        Output file path.

    Returns
    -------
    fig, axes
    """
    from cpfc_simulation.analysis.orbital_lifetime import smad_lifetime_estimate
    from cpfc_simulation.config import MASS_SAT, A_STOWED

    if mass is None:
        mass = MASS_SAT

    setup_publication_style()

    with h5py.File(h5_file, 'r') as f:
        h_range = f['h_range_km'][:]
        inc_range = f['inc_range_deg'][:]
        F107_range = f['F107_range'][:]
        Delta_y_max = f['Delta_y_max'][:]
        chaos_exists = f['chaos_exists'][:]

    # Select 3 F10.7 panels
    f107_indices = [0, len(F107_range) // 2, len(F107_range) - 1]
    F107_values = F107_range[f107_indices]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    # Precompute lifetime grid for each F10.7 (over altitude only;
    # inclination doesn't affect SMAD lifetime significantly)
    for panel, (f107_idx, f107_val) in enumerate(zip(f107_indices, F107_values)):
        ax = axes[panel]

        data = Delta_y_max[:, :, f107_idx]

        # Safety boundary heatmap
        im = ax.pcolormesh(inc_range, h_range, data, cmap='viridis',
                           shading='auto')
        plt.colorbar(im, ax=ax, label=r'$\Delta y_{\max}$ [m]')

        # Chaos boundary contour
        if chaos_exists is not None:
            chaos_data = chaos_exists[:, :, f107_idx].astype(float)
            ax.contour(inc_range, h_range, chaos_data, levels=[0.5],
                       colors='white', linewidths=2, linestyles='--')

        # Lifetime contours (computed over altitude grid, constant across inc)
        lifetimes_1d = np.array([
            smad_lifetime_estimate(h, Cd=Cd, A_cross=A_STOWED, mass=mass,
                                   F107=f107_val)
            for h in h_range
        ])
        # Broadcast to 2D (altitude x inclination)
        lifetime_2d = np.tile(lifetimes_1d[:, np.newaxis], (1, len(inc_range)))

        # Overlay lifetime contours
        contour_levels = [30, 60, 90, 180, 365]
        contour_levels = [lv for lv in contour_levels
                          if lifetime_2d.min() < lv < lifetime_2d.max()]
        if contour_levels:
            cs = ax.contour(inc_range, h_range, lifetime_2d,
                            levels=contour_levels, colors='red',
                            linewidths=1.5, linestyles='-')
            ax.clabel(cs, fmt='%d d', fontsize=8, colors='red')

        # Mission feasibility boundary
        mission_levels = [mission_days]
        if lifetime_2d.min() < mission_days < lifetime_2d.max():
            cs_m = ax.contour(inc_range, h_range, lifetime_2d,
                              levels=mission_levels, colors='yellow',
                              linewidths=3, linestyles='-')
            ax.clabel(cs_m, fmt=f'{mission_days:.0f} d limit', fontsize=9,
                      colors='yellow')

        ax.set_xlabel('Inclination [deg]')
        if panel == 0:
            ax.set_ylabel('Altitude [km]')
        ax.set_title(f'F10.7 = {f107_val:.0f} SFU')

    fig.suptitle('Melnikov Safety Boundary with Orbital Lifetime Contours\n'
                 '(red contours = estimated lifetime in days; '
                 'yellow = mission feasibility limit)',
                 fontsize=13, y=1.04)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig, axes
