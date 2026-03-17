# CPFC: Chaotic Passive Formation Control Framework

A physics-driven simulation framework for propellant-free LEO satellite formation keeping using differential drag, natural orbital perturbations (J2/J3/J4), and chaos-assisted attitude control.

## Novel Contribution

This is the first unified framework that simultaneously:

1. **Couples attitude chaos to orbital maneuvering** -- uses chaotic dynamics in attitude phase space (Doroshin/Aslanov methodology) to deliberately generate the exact differential drag coefficient needed for formation control.
2. **Treats J2 and atmospheric density as designed inputs** -- applies Poincare map analysis to the Schweighart-Sedwick relative motion phase space, analogous to how Koon-Lo-Marsden-Ross use manifold tubes in CR3BP.
3. **Introduces the CAPR (Chaos-Assisted Passive Reconfiguration) Law** -- replaces thrusters entirely with a binary drag plate deployment sequence derived from the chaotic attractor's phase-space geometry.
4. **Provides the Melnikov Formation Safety Boundary** -- an analytic surface in (altitude, inclination, solar flux) space that separates stable formation-keeping from uncontrolled drift.

## Installation

**Requirements:** Python 3.10+, Anaconda recommended.

```bash
pip install numpy scipy matplotlib astropy numba h5py pandas tqdm nrlmsise00
```

## Quick Start

```bash
# 1-day test at 320 km (strongest differential drag)
python -m cpfc_simulation.main --days 1 --alt 320 --no-safety-boundary

# Full 7-day mission with safety boundary map
python -m cpfc_simulation.main --days 7 --alt 320

# Custom altitude (300-450 km range)
python -m cpfc_simulation.main --days 7 --alt 350
```

Results (figures + HDF5 data) are saved to `results/`.

## Package Structure

```
cpfc_simulation/
├── main.py                          # Pipeline orchestrator (v2: coupled dynamics)
├── config.py                        # Physical constants (IERS 2010) & mission parameters
├── dynamics/
│   ├── orbit_propagator.py          # High-fidelity ECI propagator (J2+J3+J4+drag+SRP)
│   ├── relative_motion.py           # Corrected Schweighart-Sedwick (Traub et al. 2025)
│   ├── attitude_dynamics.py         # Euler equations + gravity gradient + pitch-only mode
│   ├── perturbations.py             # Zonal harmonics, NRLMSISE-00 drag, SRP with shadow
│   ├── aerodynamic_model.py         # Sentman free-molecular flow Cd for 3U CubeSat
│   └── coupled_system.py           # Coupled attitude-orbital dynamics (closes the loop)
├── chaos/
│   ├── melnikov.py                  # Melnikov integral & chaos prediction
│   ├── poincare_map.py              # Poincare sections & invariant manifold tracing
│   ├── lyapunov.py                  # MLE (Wolf 1985) & FTLE field computation
│   ├── attractor_library.py         # Doroshin Types 1-5, Lorenz, Rossler
│   └── capr_law.py                  # Chaos-Assisted Passive Reconfiguration Law
├── formation/
│   ├── formation_geometry.py        # PCO/GCO definitions for N-satellite formations
│   ├── phase_plane_analysis.py      # CW & SS-corrected ellipses, accessible families
│   └── safety_boundary.py           # Melnikov safety map (novel, HDF5 output)
├── control/
│   ├── drag_plate_scheduler.py      # Binary ON/OFF with S-NET physical constraints
│   └── benchmark_controllers.py     # LP, convex optimization, constraint tightening, oracle
├── analysis/
│   ├── metrics.py                   # RMS error, P99, delta-V equivalent, lifetime
│   └── monte_carlo.py              # Atmospheric uncertainty (F10.7, Ap) MC analysis
├── visualization/
│   ├── phase_portraits.py          # Attitude phase space & Poincare sections
│   ├── formation_plots.py          # 3D formation geometry & evolution
│   └── safety_maps.py              # Melnikov boundary heatmaps
└── data/
    └── space_weather/              # Downloaded F10.7, Ap index files
```

## Simulation Pipeline

The `main.py` orchestrator runs five steps:

| Step | Description | Output |
|------|-------------|--------|
| 1 | Precompute Melnikov chaos boundaries, heteroclinic orbits, attractor survey | `melnikov_spectrum.png`, `attitude_phase_portrait.png` |
| 2-3 | Run coupled formation simulation with CAPR + 5 benchmarks + uncontrolled baseline | Controller comparison table |
| 4 | Demonstrate attractor-averaged Cd mechanism (chaos -> drag coupling proof) | `attractor_CdA_mechanism.png`, `CdA_time_series.png` |
| 5 | Melnikov safety boundary parameter sweep (250-600 km, 0-98 deg, 3 solar flux levels) | `safety_boundary_map.h5`, `safety_boundary_heatmap.png` |
| 6 | Publication figures: error comparison, formation evolution, attitude/CdA histories | 10+ figures |

## Physical Models

### Orbit Propagation
- **Gravity:** J2, J3, J4 zonal harmonics (EGM2008 coefficients)
- **Atmosphere:** NRLMSISE-00 empirical density model with real F10.7 and Ap inputs
- **Drag:** Sentman (1961) / Schaaf-Chambre free-molecular flow model (Kn >> 10 above 120 km)
- **SRP:** Cannonball model with cylindrical Earth shadow
- **Integration:** RK45 adaptive, rtol=1e-10, atol=1e-12

### Relative Motion
- Schweighart-Sedwick J2-corrected equations with the Traub et al. (2025) secular drift correction term
- Validated: closed-form vs numerical agreement to ~1e-12 m per orbit

### Attitude Dynamics
- Full 3-axis Euler equations with 3-2-1 kinematic sequence
- Gravity gradient torque (Wie 2008)
- Aerodynamic restoring torque (Aslanov & Sizov 2021)
- Simplified pitch-only mode for chaos studies

### Chaos Analysis
- Melnikov integral via numerical quadrature on heteroclinic orbits
- Wolf et al. (1985) maximum Lyapunov exponent algorithm
- Finite-Time Lyapunov Exponent (FTLE) fields for Lagrangian Coherent Structures
- Five Doroshin & Elisov (2024) attractor types adapted for CubeSat drag-plate excitation

## Mission Parameters (Default)

| Parameter | Value |
|-----------|-------|
| Satellite | 3U CubeSat, 4.0 kg |
| Altitude | 450 km (LEO) |
| Inclination | 97.5 deg (sun-synchronous) |
| Formation | 4 satellites, 500 m PCO radius |
| Drag plates | 2 x 30 cm x 10 cm deployable panels |
| Mission duration | 30 days |
| Epoch | 2024-03-20 12:00 UTC |

## Benchmark Controllers

| Controller | Method | Reference |
|-----------|--------|-----------|
| CAPR | Chaos-assisted passive reconfiguration | This work |
| LP | Linear programming differential drag | Small Satellite Constellation Separation |
| Convex | Convex optimization with binary relaxation | Convex Optimization of Relative Orbit Maneuvers |
| Constraint Tightening | Safety-margin tightened control | Advances in Space Research (2024) |
| Oracle Thruster | Ideal continuous thrust (upper bound) | -- |

## Key References

- Traub, C. et al. (2025). *Operationalizing differential drag.* Acta Astronautica 234, 742-753.
- Schweighart, S. & Sedwick, R. (2002). *High-fidelity linearized J2 model for satellite formation flight.* JGCD 25(6).
- Aslanov, V. & Sizov, D. (2021). *3U CubeSat aerodynamic drag and attitude.* Acta Astronautica 189, 310-320.
- Doroshin, A. & Elisov, N. (2024). *New chaotic attractors in Euler attitude equations.*
- Koon, W.S. et al. (2000). *Dynamical Systems, the Three-Body Problem and Space Mission Design.*
- Sentman, L.H. (1961). *Free molecule flow theory.* DTIC AD0265409.

## License

Research use. Contact the author for licensing.
