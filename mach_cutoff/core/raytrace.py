"""Adaptive RK4 ray integrator."""

from __future__ import annotations

import numpy as np


def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("zero-length vector")
    return v / n


def integrate_ray(
    r0,
    dir0,
    field,
    ds=0.01,
    steps=1000,
    *,
    adaptive=True,
    tol=1e-5,
    min_step=None,
    max_step=None,
    domain_bounds=None,
    surfaces=None,
    stop_on_exit=True,
):
    """Integrate a ray through a scalar refractive-index field."""
    r = np.asarray(r0, dtype=float)
    u0 = _normalize(dir0)
    n0 = field.n(r)
    p = n0 * u0

    state = np.empty(6, dtype=float)
    state[0:3] = r
    state[3:6] = p

    def deriv(sstate):
        rr = sstate[0:3]
        pp = sstate[3:6]
        nn = field.n(rr)
        drds = pp / nn
        dpds = field.grad_n(rr)
        return np.concatenate([drds, dpds])

    def rk4_step(sstate, step):
        k1 = deriv(sstate)
        k2 = deriv(sstate + 0.5 * step * k1)
        k3 = deriv(sstate + 0.5 * step * k2)
        k4 = deriv(sstate + step * k3)
        return sstate + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def adaptive_step(sstate, step):
        full = rk4_step(sstate, step)
        half = rk4_step(rk4_step(sstate, 0.5 * step), 0.5 * step)
        err = np.linalg.norm(half[0:3] - full[0:3])
        return half, err

    bounds = None
    if domain_bounds is not None:
        bounds = np.asarray(domain_bounds, dtype=float)
        if bounds.shape != (3, 2):
            raise ValueError("domain_bounds must have shape (3, 2)")

    surface_tests = list(surfaces or [])

    def should_stop(pos):
        if bounds is not None and stop_on_exit:
            if np.any(pos < bounds[:, 0]) or np.any(pos > bounds[:, 1]):
                return True
        for surface in surface_tests:
            if surface(pos):
                return True
        return False

    if not adaptive:
        traj = np.zeros((steps + 1, 3), dtype=float)
        traj[0] = state[0:3].copy()
        valid = 1
        for _ in range(steps):
            state = rk4_step(state, ds)
            traj[valid] = state[0:3]
            valid += 1
            if should_stop(state[0:3]):
                return traj[:valid]
        return traj

    traj = [state[0:3].copy()]
    h = float(ds)
    min_h = ds / 32.0 if min_step is None else float(min_step)
    max_h = ds * 5.0 if max_step is None else float(max_step)
    h = np.clip(h, min_h, max_h)

    for _ in range(steps):
        next_state, err = adaptive_step(state, h)
        if err > tol and h > min_h:
            h = max(h * 0.5, min_h)
            continue

        state = next_state
        traj.append(state[0:3].copy())
        if should_stop(state[0:3]):
            break

        if err < tol / 4.0 and h < max_h:
            h = min(h * 2.0, max_h)

    return np.vstack(traj)
