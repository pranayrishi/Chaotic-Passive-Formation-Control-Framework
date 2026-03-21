# config.py — ALL UNITS SI UNLESS STATED
import numpy as np
from datetime import datetime

# ─── Earth Parameters (IERS 2010 + EGM2008) ───────────────────────────────────
MU_EARTH       = 3.986004418e14   # [m^3/s^2] Earth gravitational parameter
R_EARTH        = 6.3781366e6      # [m] Earth mean equatorial radius
J2             = 1.08262668e-3    # dimensionless J2 zonal harmonic (EGM2008)
J3             = -2.53265648e-6   # dimensionless J3 zonal harmonic
J4             = -1.61962159e-6   # dimensionless J4 zonal harmonic
OMEGA_EARTH    = 7.2921150e-5     # [rad/s] Earth rotation rate
C_LIGHT        = 2.99792458e8     # [m/s] speed of light
AU             = 1.495978707e11   # [m] 1 Astronomical Unit

# ─── Solar Radiation Pressure ─────────────────────────────────────────────────
P_SRP          = 4.56e-6          # [N/m^2] solar radiation pressure at 1 AU
SOLAR_FLUX_AU  = 1361.0           # [W/m^2] solar constant

# ─── 3U CubeSat Physical Parameters (based on Aslanov 2021, Table 1) ─────────
MASS_SAT       = 4.0              # [kg] wet mass (typical 3U)
L_X            = 0.1              # [m] body dimension along roll axis
L_Y            = 0.1              # [m] body dimension along pitch axis
L_Z            = 0.3405           # [m] body dimension along yaw axis (3U stack)
IXX            = (1.0/12.0)*MASS_SAT*(L_Y**2 + L_Z**2)
IYY            = (1.0/12.0)*MASS_SAT*(L_X**2 + L_Z**2)
IZZ            = (1.0/12.0)*MASS_SAT*(L_X**2 + L_Y**2)

# Drag plate geometry
PANEL_LENGTH   = 0.30             # [m] deployable panel length
PANEL_WIDTH    = 0.10             # [m] deployable panel width
A_STOWED       = L_X * L_Y       # [m^2] minimum cross-section
A_DEPLOYED     = PANEL_LENGTH * PANEL_WIDTH * 2 + A_STOWED  # [m^2] both panels out
A_REF          = A_STOWED
CR             = 1.3              # solar radiation reflectivity coefficient

# ─── Formation Mission Parameters ─────────────────────────────────────────────
N_SATELLITES   = 4
ALT_NOMINAL    = 450e3            # [m] 450 km LEO
INC_NOMINAL    = 51.6             # [deg] ISS-like (stable SS: (1+2c) > 0)
RAAN_NOMINAL   = 0.0              # [deg]
FORMATION_RADIUS = 500.0          # [m] PCO formation radius
SEPARATION_TARGET = 500.0         # [m] in-track separation target

# ─── Chaos Control Parameters ─────────────────────────────────────────────────
DE_POPULATION  = 50
DE_MAXITER     = 500
DE_MUTATION    = 0.8
DE_RECOMBINATION = 0.7

# ─── Atmospheric Model Parameters ─────────────────────────────────────────────
F107_NOMINAL   = 150.0
F107A_NOMINAL  = 150.0
AP_NOMINAL     = 10
F107_RANGE     = np.arange(70, 250, 20)
AP_RANGE       = np.array([0, 4, 9, 15, 27, 48, 80, 132])

# ─── Simulation Parameters ────────────────────────────────────────────────────
DT_INTEGRATOR  = 1.0             # [s] max integration timestep
T_MISSION      = 30 * 86400.0    # [s] 30-day mission
T_ORBIT        = 2*np.pi*np.sqrt((R_EARTH + ALT_NOMINAL)**3 / MU_EARTH)

# ─── Epoch ────────────────────────────────────────────────────────────────────
EPOCH          = datetime(2024, 3, 20, 12, 0, 0)  # Spring equinox 2024
