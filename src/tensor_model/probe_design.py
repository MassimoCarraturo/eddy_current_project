"""
Probe-configuration design ("rosette" design) -- Step 3.

Given the elastoresistivity model and the directional-probe model from Step 2,
this module finds probe configurations (sets of sensing directions) that make
the strain inversion well-posed: full rank (all resolvable components covered)
and a small condition number (noise-robust).

Two design questions are answered:

  * optimize_inplane_rosette : the best set of N in-plane sensing angles -- the
    direct analogue of a mechanical strain-gauge rosette. For N = 3 the optimum
    is the evenly spaced "delta rosette" (0/60/120 deg).

  * optimize_probe_set : the best set of N fully 3-D sensing directions
    (azimuth + tilt), used when out-of-plane access is available.

Fundamental limit
-----------------
The strain map is A = M . K, where M is the directional-probe matrix and K the
elastoresistivity matrix. No probe set can resolve more strain components than
rank(K): if the material response is purely volumetric (kappa_long == kappa_trans,
K has rank 1), every configuration resolves a single DOF -- directional
diversity helps only to the extent the elastoresistive response is anisotropic.
``max_achievable_rank`` reports this ceiling.
"""

import numpy as np
from scipy.optimize import minimize

from .observability import ObservabilityAnalysis, direction


def design_metrics(A: np.ndarray) -> dict:
    """Rank, condition number and D-optimality of a measurement matrix."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    s = np.linalg.svd(A, compute_uv=False)
    smax = s[0] if s.size else 0.0
    tol = max(A.shape) * np.finfo(float).eps * smax
    rank = int((s > tol).sum())
    cond = float(s[0] / s[rank - 1]) if rank > 0 else np.inf
    d_opt = float(np.prod(s[:rank] ** 2)) if rank > 0 else 0.0  # info-ellipsoid volume
    return {"rank": rank, "condition_number": cond,
            "singular_values": s, "d_optimality": d_opt}


def evaluate_configuration(directions, model, sigma0: float = 1.0) -> dict:
    """Design metrics for an explicit set of sensing directions."""
    A = ObservabilityAnalysis.from_directions(directions).strain_map(model, sigma0)
    return design_metrics(A)


def max_achievable_rank(model, allow_tilt: bool, sigma0: float = 1.0) -> int:
    """Upper bound on resolvable strain DOF for a probe family.

    Built from a dense candidate set of directions; equals rank(K) when tilt is
    allowed, or the in-plane-reachable rank otherwise.
    """
    az = np.linspace(0, 170, 18)
    if allow_tilt:
        tilts = np.linspace(10, 90, 9)
        dirs = [direction(a, t) for a in az for t in tilts]
    else:
        dirs = [direction(a, 90.0) for a in az]
    return evaluate_configuration(dirs, model, sigma0)["rank"]


# -- Internal helpers --------------------------------------------------------

def _directions_from_params(params, allow_tilt):
    if allow_tilt:
        p = np.asarray(params, dtype=float).reshape(-1, 2)
        return [direction(np.degrees(a), np.degrees(t)) for a, t in p]
    return [direction(np.degrees(a), 90.0) for a in np.asarray(params, dtype=float)]


def _objective(params, model, sigma0, allow_tilt, target_rank):
    dirs = _directions_from_params(params, allow_tilt)
    A = ObservabilityAnalysis.from_directions(dirs).strain_map(model, sigma0)
    m = design_metrics(A)
    if m["rank"] < target_rank:
        return 1.0e6 + (target_rank - m["rank"]) * 1.0e5
    return np.log10(m["condition_number"])


def _optimize(model, n_probes, allow_tilt, sigma0, target_rank,
              n_restarts, seed):
    rng = np.random.default_rng(seed)
    best = None
    n_par = 2 * n_probes if allow_tilt else n_probes
    for _ in range(n_restarts):
        x0 = rng.uniform(0.0, np.pi, size=n_par)
        res = minimize(_objective, x0,
                       args=(model, sigma0, allow_tilt, target_rank),
                       method="Nelder-Mead",
                       options={"maxiter": 6000, "xatol": 1e-5, "fatol": 1e-5})
        dirs = _directions_from_params(res.x, allow_tilt)
        m = evaluate_configuration(dirs, model, sigma0)
        if m["rank"] < target_rank:
            continue
        if best is None or m["condition_number"] < best["condition_number"]:
            angles = np.asarray(res.x, dtype=float).reshape(-1, 2) if allow_tilt \
                else np.asarray(res.x, dtype=float).reshape(-1, 1)
            best = {"directions": dirs,
                    "angles_deg": np.degrees(angles % np.pi),
                    **m}
    return best


def optimize_inplane_rosette(model, n_probes: int = 3, sigma0: float = 1.0,
                             n_restarts: int = 16, seed: int = 0) -> dict:
    """Optimise N in-plane sensing angles to minimise the condition number.

    Returns the best configuration found (directions, angles, metrics).
    """
    target = min(n_probes, max_achievable_rank(model, allow_tilt=False, sigma0=sigma0))
    return _optimize(model, n_probes, allow_tilt=False, sigma0=sigma0,
                     target_rank=target, n_restarts=n_restarts, seed=seed)


def optimize_probe_set(model, n_probes: int = 6, sigma0: float = 1.0,
                       n_restarts: int = 20, seed: int = 0) -> dict:
    """Optimise N fully 3-D sensing directions (azimuth + tilt).

    Returns the best configuration found that reaches the achievable rank.
    """
    target = min(n_probes, max_achievable_rank(model, allow_tilt=True, sigma0=sigma0))
    return _optimize(model, n_probes, allow_tilt=True, sigma0=sigma0,
                     target_rank=target, n_restarts=n_restarts, seed=seed)
