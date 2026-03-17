"""Poincare section construction and invariant manifold tracing."""
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.cluster import KMeans


def compute_poincare_map(state0, t_max, coupled_eom, section_idx=3, section_val=0.0,
                         section_sign=1):
    """
    Integrate full system and record Poincare section crossings.

    Parameters
    ----------
    state0 : array-like
        Initial state vector.
    t_max : float
        Maximum integration time.
    coupled_eom : callable
        Equations of motion f(t, state) -> dstate/dt.
    section_idx : int
        Index of the state variable defining the section (e.g., 3 for theta).
    section_val : float
        Value of state[section_idx] at the section plane.
    section_sign : int
        +1 for positive crossings, -1 for negative crossings.

    Returns
    -------
    crossings : ndarray, shape (N_crossings, n_dim)
        State vectors at each crossing.
    crossing_times : ndarray, shape (N_crossings,)
        Times of each crossing.
    full_sol : OdeSolution
        The full integration solution.
    """
    n_dim = len(state0)

    def section_event(t, state):
        return state[section_idx] - section_val

    section_event.terminal = False
    section_event.direction = section_sign

    sol = solve_ivp(coupled_eom, [0, t_max], state0, method='RK45',
                    rtol=1e-10, atol=1e-12, events=section_event,
                    max_step=1.0, dense_output=True)

    crossing_times = sol.t_events[0]
    if len(crossing_times) == 0:
        return np.empty((0, n_dim)), np.array([]), sol

    crossings = np.array([sol.sol(tc) for tc in crossing_times])
    return crossings, crossing_times, sol


def find_fixed_points(crossings, n_clusters=5):
    """
    Identify approximate fixed points from Poincare crossings using k-means clustering.

    Parameters
    ----------
    crossings : ndarray, shape (N, n_dim)
        Poincare section crossings.
    n_clusters : int
        Number of clusters to try.

    Returns
    -------
    fixed_points : ndarray, shape (n_found, n_dim)
        Estimated fixed points (cluster centers with tight clusters).
    cluster_labels : ndarray, shape (N,)
        Cluster label for each crossing.
    """
    if len(crossings) < n_clusters:
        n_clusters = max(1, len(crossings))

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(crossings)
    centers = kmeans.cluster_centers_

    # Filter: keep clusters whose members are tightly grouped (std < median std)
    stds = []
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() > 1:
            stds.append(np.mean(np.std(crossings[mask], axis=0)))
        else:
            stds.append(np.inf)
    stds = np.array(stds)

    # Keep clusters tighter than twice the median spread
    median_std = np.median(stds[np.isfinite(stds)]) if np.any(np.isfinite(stds)) else np.inf
    tight_mask = stds < 2.0 * median_std
    fixed_points = centers[tight_mask]

    return fixed_points, labels


def compute_jacobian_at_fixed_point(fixed_point, coupled_eom, section_idx,
                                     section_val=0.0, section_sign=1, delta=1e-6):
    """
    Numerical Jacobian of the Poincare map at a fixed point using finite differences.

    For each state component i (excluding section_idx):
        1. Perturb fixed_point[i] by +delta and -delta
        2. Integrate one full return to the section
        3. Compute partial derivative via central difference

    Parameters
    ----------
    fixed_point : ndarray
        State at the fixed point (on the section).
    coupled_eom : callable
        Equations of motion.
    section_idx : int
        Section variable index.
    section_val : float
        Section value.
    section_sign : int
        Crossing direction.
    delta : float
        Finite difference step.

    Returns
    -------
    J : ndarray, shape (n_reduced, n_reduced)
        Jacobian of the Poincare return map in reduced coordinates
        (all dimensions except section_idx).
    """
    n_dim = len(fixed_point)
    # Reduced indices: all except section_idx
    reduced_idx = [i for i in range(n_dim) if i != section_idx]
    n_red = len(reduced_idx)

    def section_event(t, state):
        return state[section_idx] - section_val

    section_event.terminal = True
    section_event.direction = section_sign

    def integrate_one_return(state0):
        """Integrate until the next section crossing after a short initial phase."""
        # First integrate a short time to move away from the section
        t_short = 0.1
        sol0 = solve_ivp(coupled_eom, [0, t_short], state0, method='RK45',
                         rtol=1e-11, atol=1e-13, max_step=0.5)
        state_off = sol0.y[:, -1]

        # Now integrate until next crossing
        t_max_return = 1e5
        sol = solve_ivp(coupled_eom, [0, t_max_return], state_off, method='RK45',
                        rtol=1e-11, atol=1e-13, events=section_event, max_step=1.0)
        if len(sol.t_events[0]) > 0:
            tc = sol.t_events[0][0]
            return sol.sol(tc) if sol.sol is not None else sol.y_events[0][0]
        # Fallback: return last state
        return sol.y[:, -1]

    J = np.zeros((n_red, n_red))

    for j_col, idx_j in enumerate(reduced_idx):
        # +delta perturbation
        state_plus = fixed_point.copy()
        state_plus[idx_j] += delta
        result_plus = integrate_one_return(state_plus)

        # -delta perturbation
        state_minus = fixed_point.copy()
        state_minus[idx_j] -= delta
        result_minus = integrate_one_return(state_minus)

        for i_row, idx_i in enumerate(reduced_idx):
            J[i_row, j_col] = (result_plus[idx_i] - result_minus[idx_i]) / (2 * delta)

    return J


def trace_invariant_manifold(fixed_point, jacobian, coupled_eom, section_idx=3,
                              section_val=0.0, section_sign=1,
                              n_points=100, eps=1e-4, direction='unstable'):
    """
    Trace stable or unstable invariant manifolds of a fixed point.

    Algorithm:
    1. Compute eigenvectors of the Jacobian.
    2. For unstable manifold: use eigenvectors with |eigenvalue| > 1.
       For stable manifold: use eigenvectors with |eigenvalue| < 1.
    3. Displace initial conditions along chosen eigenvectors by ±eps.
    4. Integrate forward (unstable) or backward (stable) in time.

    Parameters
    ----------
    fixed_point : ndarray
        Fixed point on the Poincare section.
    jacobian : ndarray
        Jacobian of the Poincare map at the fixed point.
    coupled_eom : callable
        Equations of motion.
    section_idx : int
        Section variable index.
    section_val : float
        Section value.
    section_sign : int
        Crossing direction.
    n_points : int
        Number of initial displacements along each eigenvector.
    eps : float
        Maximum displacement magnitude.
    direction : str
        'unstable' or 'stable'.

    Returns
    -------
    manifold_trajectories : list of ndarray
        Each element is shape (N_steps, n_dim), the trajectory from one IC.
    eigenvalues : ndarray
        Eigenvalues of the Jacobian (for diagnostics).
    """
    n_dim = len(fixed_point)
    reduced_idx = [i for i in range(n_dim) if i != section_idx]

    eigenvalues, eigenvectors = np.linalg.eig(jacobian)

    # Select relevant eigenvectors
    if direction == 'unstable':
        mask = np.abs(eigenvalues) > 1.0
        t_sign = 1.0   # integrate forward
    else:
        mask = np.abs(eigenvalues) < 1.0
        t_sign = -1.0  # integrate backward

    if not np.any(mask):
        # No eigenvalues satisfy the criterion; use the closest to threshold
        if direction == 'unstable':
            idx_best = np.argmax(np.abs(eigenvalues))
        else:
            idx_best = np.argmin(np.abs(eigenvalues))
        mask = np.zeros(len(eigenvalues), dtype=bool)
        mask[idx_best] = True

    selected_evecs = eigenvectors[:, mask].real
    T_manifold = 5000.0  # integration time for manifold tracing

    manifold_trajectories = []

    for col in range(selected_evecs.shape[1]):
        evec_reduced = selected_evecs[:, col]
        evec_reduced /= np.linalg.norm(evec_reduced)

        # Full-dimensional eigenvector
        evec_full = np.zeros(n_dim)
        for k, ri in enumerate(reduced_idx):
            evec_full[ri] = evec_reduced[k]

        eps_values = np.linspace(-eps, eps, n_points)
        eps_values = eps_values[eps_values != 0.0]  # skip zero displacement

        for e in eps_values:
            ic = fixed_point + e * evec_full
            sol = solve_ivp(coupled_eom, [0, t_sign * T_manifold], ic,
                            method='RK45', rtol=1e-10, atol=1e-12,
                            max_step=1.0)
            traj = sol.y.T  # shape (N_steps, n_dim)
            manifold_trajectories.append(traj)

    return manifold_trajectories, eigenvalues


class PoincareSectionAnalyzer:
    """Bundles Poincare section computation, fixed point finding, and manifold tracing."""

    def __init__(self, coupled_eom, section_idx=3, section_val=0.0, section_sign=1):
        self.coupled_eom = coupled_eom
        self.section_idx = section_idx
        self.section_val = section_val
        self.section_sign = section_sign

        self.crossings = None
        self.crossing_times = None
        self.full_sol = None
        self.fixed_points = None
        self.jacobians = {}
        self.manifolds = {}

    def compute_section(self, state0, t_max):
        """Compute the Poincare section crossings."""
        self.crossings, self.crossing_times, self.full_sol = compute_poincare_map(
            state0, t_max, self.coupled_eom,
            section_idx=self.section_idx,
            section_val=self.section_val,
            section_sign=self.section_sign
        )
        return self.crossings, self.crossing_times

    def find_fixed_points(self, n_clusters=5):
        """Find fixed points from stored crossings."""
        if self.crossings is None or len(self.crossings) == 0:
            raise ValueError("No crossings computed. Call compute_section first.")
        self.fixed_points, labels = find_fixed_points(self.crossings, n_clusters)
        return self.fixed_points, labels

    def compute_jacobian(self, fp_index=0, delta=1e-6):
        """Compute Jacobian at a specific fixed point."""
        if self.fixed_points is None or len(self.fixed_points) == 0:
            raise ValueError("No fixed points found. Call find_fixed_points first.")
        fp = self.fixed_points[fp_index]
        J = compute_jacobian_at_fixed_point(
            fp, self.coupled_eom, self.section_idx,
            self.section_val, self.section_sign, delta
        )
        self.jacobians[fp_index] = J
        return J

    def trace_manifold(self, fp_index=0, direction='unstable', n_points=100, eps=1e-4):
        """Trace invariant manifold at a specific fixed point."""
        if fp_index not in self.jacobians:
            self.compute_jacobian(fp_index)
        fp = self.fixed_points[fp_index]
        J = self.jacobians[fp_index]
        trajs, evals = trace_invariant_manifold(
            fp, J, self.coupled_eom,
            section_idx=self.section_idx,
            section_val=self.section_val,
            section_sign=self.section_sign,
            n_points=n_points, eps=eps, direction=direction
        )
        key = (fp_index, direction)
        self.manifolds[key] = trajs
        return trajs, evals

    def full_analysis(self, state0, t_max, n_clusters=5, trace_manifolds=True,
                      n_manifold_points=50, eps=1e-4):
        """
        Run full Poincare analysis pipeline:
        1. Compute section crossings
        2. Find fixed points
        3. Compute Jacobians
        4. Trace stable and unstable manifolds

        Returns
        -------
        results : dict
            Dictionary with crossings, fixed_points, jacobians, manifolds.
        """
        self.compute_section(state0, t_max)
        if len(self.crossings) == 0:
            return {'crossings': self.crossings, 'fixed_points': np.array([]),
                    'jacobians': {}, 'manifolds': {}}

        fps, labels = self.find_fixed_points(n_clusters)

        if trace_manifolds and len(fps) > 0:
            for i in range(len(fps)):
                try:
                    self.compute_jacobian(i)
                    self.trace_manifold(i, direction='unstable',
                                        n_points=n_manifold_points, eps=eps)
                    self.trace_manifold(i, direction='stable',
                                        n_points=n_manifold_points, eps=eps)
                except Exception:
                    # Some fixed points may not have clean manifolds
                    pass

        return {
            'crossings': self.crossings,
            'crossing_times': self.crossing_times,
            'fixed_points': self.fixed_points,
            'jacobians': dict(self.jacobians),
            'manifolds': dict(self.manifolds),
        }
