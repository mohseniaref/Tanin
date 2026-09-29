"""Noise covariance models and least-squares variance-component estimation."""
import numpy as np
import xarray as xr
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import lsq_linear
from .models import design_matrix
from .time import to_decimal_days

def white_covariance(t): return np.eye(len(t))

def random_walk_covariance(t):
    u = np.asarray(t, dtype=float) - np.min(t)
    return np.minimum.outer(u, u)

def flicker_covariance(t, n_frequencies=256):
    """Finite-band numerical approximation to a unit 1/f covariance."""
    t = np.asarray(t, dtype=float); span = max(np.ptp(t), np.finfo(float).eps)
    gaps = np.diff(np.sort(t)); gaps = gaps[gaps > 0]
    high = 0.5 / (np.min(gaps) if gaps.size else span); low = 1.0 / span
    if high <= low: high = 2 * low
    freq = np.geomspace(low, high, n_frequencies)
    dt = np.abs(t[:, None] - t[None, :])
    q = np.trapz(np.cos(2*np.pi*freq[:, None, None]*dt) / freq[:, None, None], np.log(freq), axis=0)
    q /= max(float(np.mean(np.diag(q))), np.finfo(float).eps)
    return (q + q.T) / 2

def covariance_components(t, components=("white", "flicker", "random_walk")):
    builders = {"white": white_covariance, "flicker": flicker_covariance, "random_walk": random_walk_covariance}
    unknown = set(components) - set(builders)
    if unknown: raise ValueError(f"unknown noise components: {sorted(unknown)}")
    return {name: builders[name](t) for name in components}

def _projection(X, C):
    cf = cho_factor(C, check_finite=False)
    CiX = cho_solve(cf, X, check_finite=False)
    return cho_solve(cf, np.eye(len(C)), check_finite=False) - CiX @ np.linalg.pinv(X.T @ CiX) @ CiX.T

def estimate_noise_vce(y, t=None, components=("white", "flicker", "random_walk"), trend=True,
                       non_negative=True, max_iter=20, tolerance=1e-5, initial=None):
    """Estimate covariance amplitudes using LS-VCE / optional NNLS-VCE.

    The functional model is an intercept plus an optional centered trend.
    Components are amplitudes multiplying unit cofactor matrices.
    """
    y = np.asarray(y, dtype=float); t = np.arange(y.size, dtype=float) if t is None else to_decimal_days(t)
    mask = np.isfinite(y) & np.isfinite(t); y, t = y[mask], t[mask]
    if len(np.unique(t)) != len(t): raise ValueError("duplicate epochs are not supported")
    X, _ = design_matrix(t, periods=(), trend=trend); names = list(components)
    if len(y) <= X.shape[1] + len(names): raise ValueError("time series is too short for VCE")
    Q = covariance_components(t, names)
    sigma = np.ones(len(names)) if initial is None else np.asarray(initial, dtype=float).copy()
    if sigma.shape != (len(names),) or np.any(sigma < 0): raise ValueError("initial must be non-negative per component")
    history = []
    for iteration in range(max_iter):
        C = sum(sigma[i] * Q[name] for i, name in enumerate(names))
        C += np.eye(len(C)) * max(np.trace(C) / len(C), 1e-12) * 1e-10
        P = _projection(X, C); beta = np.linalg.lstsq(X, y, rcond=None)[0]; v = y - X @ beta
        N = np.array([[0.5*np.trace(P @ Q[a] @ P @ Q[b]) for b in names] for a in names])
        l = np.array([0.5 * v @ P @ Q[a] @ P @ v for a in names])
        sigma_new = lsq_linear(N, l, bounds=(0, np.inf)).x if non_negative else np.linalg.lstsq(N, l, rcond=None)[0]
        history.append(sigma_new.copy())
        if np.max(np.abs(sigma_new-sigma) / np.maximum(np.abs(sigma), 1e-12)) < tolerance:
            sigma = sigma_new; break
        sigma = sigma_new
    C = sum(sigma[i] * Q[name] for i, name in enumerate(names))
    return {"components": {name: float(sigma[i]) for i, name in enumerate(names)}, "covariance": C,
            "cofactors": Q, "iterations": iteration + 1, "converged": len(history) < max_iter,
            "history": np.asarray(history)}

def estimate_noise(data, dim="time", **kwargs):
    """Estimate noise for a one-dimensional DataArray."""
    if not isinstance(data, xr.DataArray) or data.ndim != 1: raise ValueError("requires a 1-D DataArray")
    return estimate_noise_vce(data.values, data[dim].values, **kwargs)
