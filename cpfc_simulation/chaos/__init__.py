"""Chaos analysis module for CPFC simulation framework."""
from cpfc_simulation.chaos.melnikov import (
    pitch_potential,
    find_unstable_equilibrium,
    compute_heteroclinic_orbit,
    melnikov_integral,
    melnikov_chaos_boundary,
    optimal_switching_frequency,
)
from cpfc_simulation.chaos.lyapunov import (
    compute_MLE_wolf,
    compute_short_time_lyapunov,
    compute_FTLE_field,
)
from cpfc_simulation.chaos.poincare_map import (
    compute_poincare_map,
    find_fixed_points,
    compute_jacobian_at_fixed_point,
    trace_invariant_manifold,
    PoincareSectionAnalyzer,
)
from cpfc_simulation.chaos.attractor_library import (
    lorenz_system,
    rossler_system,
    doroshin_attractor_1,
    doroshin_attractor_2,
    doroshin_attractor_3,
    doroshin_attractor_4,
    doroshin_attractor_5,
    generate_attractor,
    classify_attractor,
    time_averaged_cross_section,
)
from cpfc_simulation.chaos.capr_law import CAPRController
