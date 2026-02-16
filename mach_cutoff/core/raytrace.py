"""Adaptive RK4 ray integrator."""

from __future__ import annotations

import os

import numpy as np

from .constants import WGS84_A_M, WGS84_B_M

try:
    from numba import njit

    _HAS_NUMBA = os.environ.get("MACH_CUTOFF_DISABLE_NUMBA", "").strip().lower() not in {"1", "true", "yes"}
except Exception:
    _HAS_NUMBA = False

    def njit(*_args, **_kwargs):
        def _decorator(func):
            return func

        return _decorator


def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("zero-length vector")
    return v / n


def _ground_hit_ellipsoid(local_pos, origin_ecef, east, north, up, a_m, b_m):
    p_ecef = origin_ecef + local_pos[0] * east + local_pos[1] * north + local_pos[2] * up
    lhs = (p_ecef[0] ** 2 + p_ecef[1] ** 2) / (a_m * a_m) + (p_ecef[2] ** 2) / (b_m * b_m)
    return bool(lhs <= 1.0)


@njit(cache=True, fastmath=True)
def _numba_norm3(v):
    return np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


@njit(cache=True, fastmath=True)
def _numba_normalize3(v):
    out = np.empty(3, dtype=np.float64)
    n = _numba_norm3(v)
    out[0] = v[0] / n
    out[1] = v[1] / n
    out[2] = v[2] / n
    return out


@njit(cache=True, fastmath=True)
def _numba_locate(grid, q):
    n = grid.shape[0]
    if q <= grid[0]:
        return 0, 1, 0.0
    if q >= grid[n - 1]:
        return n - 2, n - 1, 1.0

    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if q < grid[mid]:
            hi = mid
        else:
            lo = mid

    w = (q - grid[lo]) / (grid[hi] - grid[lo])
    return lo, hi, w


@njit(cache=True, fastmath=True)
def _numba_trilinear(values, x0, x1, wx, y0, y1, wy, z0, z1, wz):
    c000 = values[x0, y0, z0]
    c001 = values[x0, y0, z1]
    c010 = values[x0, y1, z0]
    c011 = values[x0, y1, z1]
    c100 = values[x1, y0, z0]
    c101 = values[x1, y0, z1]
    c110 = values[x1, y1, z0]
    c111 = values[x1, y1, z1]

    c00 = c000 * (1.0 - wx) + c100 * wx
    c01 = c001 * (1.0 - wx) + c101 * wx
    c10 = c010 * (1.0 - wx) + c110 * wx
    c11 = c011 * (1.0 - wx) + c111 * wx

    c0 = c00 * (1.0 - wy) + c10 * wy
    c1 = c01 * (1.0 - wy) + c11 * wy
    return c0 * (1.0 - wz) + c1 * wz


@njit(cache=True, fastmath=True)
def _numba_sample_n_and_grad(point, grid_x, grid_y, grid_z, n_grid, grad_x, grad_y, grad_z):
    x0, x1, wx = _numba_locate(grid_x, point[0])
    y0, y1, wy = _numba_locate(grid_y, point[1])
    z0, z1, wz = _numba_locate(grid_z, point[2])

    n_val = _numba_trilinear(n_grid, x0, x1, wx, y0, y1, wy, z0, z1, wz)
    gx = _numba_trilinear(grad_x, x0, x1, wx, y0, y1, wy, z0, z1, wz)
    gy = _numba_trilinear(grad_y, x0, x1, wx, y0, y1, wy, z0, z1, wz)
    gz = _numba_trilinear(grad_z, x0, x1, wx, y0, y1, wy, z0, z1, wz)
    return n_val, gx, gy, gz


@njit(cache=True, fastmath=True)
def _numba_ground_hit(local_pos, origin_ecef, east, north, up, a2, b2):
    x = origin_ecef[0] + local_pos[0] * east[0] + local_pos[1] * north[0] + local_pos[2] * up[0]
    y = origin_ecef[1] + local_pos[0] * east[1] + local_pos[1] * north[1] + local_pos[2] * up[1]
    z = origin_ecef[2] + local_pos[0] * east[2] + local_pos[1] * north[2] + local_pos[2] * up[2]
    return ((x * x + y * y) / a2 + (z * z) / b2) <= 1.0


@njit(cache=True, fastmath=True)
def _numba_should_stop(
    pos,
    bounds,
    has_bounds,
    stop_on_exit,
    has_ground,
    origin_ecef,
    east,
    north,
    up,
    a2,
    b2,
):
    if has_bounds and stop_on_exit:
        if (
            pos[0] < bounds[0, 0]
            or pos[0] > bounds[0, 1]
            or pos[1] < bounds[1, 0]
            or pos[1] > bounds[1, 1]
            or pos[2] < bounds[2, 0]
            or pos[2] > bounds[2, 1]
        ):
            return True
    if has_ground and _numba_ground_hit(pos, origin_ecef, east, north, up, a2, b2):
        return True
    return False


@njit(cache=True, fastmath=True)
def _numba_deriv(state, grid_x, grid_y, grid_z, n_grid, grad_x, grad_y, grad_z):
    rr = state[0:3]
    n_val, gx, gy, gz = _numba_sample_n_and_grad(rr, grid_x, grid_y, grid_z, n_grid, grad_x, grad_y, grad_z)
    out = np.empty(6, dtype=np.float64)
    out[0] = state[3] / n_val
    out[1] = state[4] / n_val
    out[2] = state[5] / n_val
    out[3] = gx
    out[4] = gy
    out[5] = gz
    return out


@njit(cache=True, fastmath=True)
def _numba_rk4_step(state, step, grid_x, grid_y, grid_z, n_grid, grad_x, grad_y, grad_z):
    k1 = _numba_deriv(state, grid_x, grid_y, grid_z, n_grid, grad_x, grad_y, grad_z)
    k2 = _numba_deriv(state + 0.5 * step * k1, grid_x, grid_y, grid_z, n_grid, grad_x, grad_y, grad_z)
    k3 = _numba_deriv(state + 0.5 * step * k2, grid_x, grid_y, grid_z, n_grid, grad_x, grad_y, grad_z)
    k4 = _numba_deriv(state + step * k3, grid_x, grid_y, grid_z, n_grid, grad_x, grad_y, grad_z)
    return state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@njit(cache=True, fastmath=True)
def _numba_adaptive_step(state, step, grid_x, grid_y, grid_z, n_grid, grad_x, grad_y, grad_z):
    full = _numba_rk4_step(state, step, grid_x, grid_y, grid_z, n_grid, grad_x, grad_y, grad_z)
    half = _numba_rk4_step(
        _numba_rk4_step(state, 0.5 * step, grid_x, grid_y, grid_z, n_grid, grad_x, grad_y, grad_z),
        0.5 * step,
        grid_x,
        grid_y,
        grid_z,
        n_grid,
        grad_x,
        grad_y,
        grad_z,
    )
    dx = half[0] - full[0]
    dy = half[1] - full[1]
    dz = half[2] - full[2]
    err = np.sqrt(dx * dx + dy * dy + dz * dz)
    return half, err


@njit(cache=True, fastmath=True)
def _numba_integrate_ray(
    r0,
    dir0,
    grid_x,
    grid_y,
    grid_z,
    n_grid,
    grad_x,
    grad_y,
    grad_z,
    ds,
    steps,
    adaptive,
    tol,
    min_h,
    max_h,
    bounds,
    has_bounds,
    stop_on_exit,
    has_ground,
    origin_ecef,
    east,
    north,
    up,
    a2,
    b2,
):
    r = np.empty(3, dtype=np.float64)
    r[0] = r0[0]
    r[1] = r0[1]
    r[2] = r0[2]
    u0 = _numba_normalize3(dir0)
    n0, _, _, _ = _numba_sample_n_and_grad(r, grid_x, grid_y, grid_z, n_grid, grad_x, grad_y, grad_z)

    state = np.empty(6, dtype=np.float64)
    state[0] = r[0]
    state[1] = r[1]
    state[2] = r[2]
    state[3] = n0 * u0[0]
    state[4] = n0 * u0[1]
    state[5] = n0 * u0[2]

    traj = np.empty((steps + 1, 3), dtype=np.float64)
    traj[0, 0] = state[0]
    traj[0, 1] = state[1]
    traj[0, 2] = state[2]
    valid = 1

    if not adaptive:
        for _ in range(steps):
            state = _numba_rk4_step(state, ds, grid_x, grid_y, grid_z, n_grid, grad_x, grad_y, grad_z)
            traj[valid, 0] = state[0]
            traj[valid, 1] = state[1]
            traj[valid, 2] = state[2]
            valid += 1
            if _numba_should_stop(
                state[0:3], bounds, has_bounds, stop_on_exit, has_ground, origin_ecef, east, north, up, a2, b2
            ):
                break
        return traj[:valid]

    h = ds
    if h < min_h:
        h = min_h
    elif h > max_h:
        h = max_h

    for _ in range(steps):
        next_state, err = _numba_adaptive_step(state, h, grid_x, grid_y, grid_z, n_grid, grad_x, grad_y, grad_z)
        if err > tol and h > min_h:
            h = max(h * 0.5, min_h)
            continue

        state = next_state
        traj[valid, 0] = state[0]
        traj[valid, 1] = state[1]
        traj[valid, 2] = state[2]
        valid += 1
        if _numba_should_stop(state[0:3], bounds, has_bounds, stop_on_exit, has_ground, origin_ecef, east, north, up, a2, b2):
            break

        if err < tol * 0.25 and h < max_h:
            h = min(h * 2.0, max_h)

    return traj[:valid]


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
    ground_origin_ecef=None,
    ground_east=None,
    ground_north=None,
    ground_up=None,
    ground_a_m=WGS84_A_M,
    ground_b_m=WGS84_B_M,
):
    """Integrate a ray through a scalar refractive-index field."""
    n_and_grad_fn = getattr(field, "n_and_grad", None)
    use_fused_query = callable(n_and_grad_fn)

    has_ground = all(
        value is not None for value in (ground_origin_ecef, ground_east, ground_north, ground_up)
    )
    ground_origin = None
    ground_e = None
    ground_n = None
    ground_u = None
    if has_ground:
        ground_origin = np.asarray(ground_origin_ecef, dtype=float).reshape(3)
        ground_e = np.asarray(ground_east, dtype=float).reshape(3)
        ground_n = np.asarray(ground_north, dtype=float).reshape(3)
        ground_u = np.asarray(ground_up, dtype=float).reshape(3)

    r = np.asarray(r0, dtype=float)
    u0 = _normalize(dir0)
    if use_fused_query:
        n0, _ = n_and_grad_fn(r)
    else:
        n0 = field.n(r)
    p = n0 * u0

    state = np.empty(6, dtype=float)
    state[0:3] = r
    state[3:6] = p

    def deriv(sstate):
        rr = sstate[0:3]
        pp = sstate[3:6]
        if use_fused_query:
            nn, dpds = n_and_grad_fn(rr)
        else:
            nn = field.n(rr)
            dpds = field.grad_n(rr)
        drds = pp / nn
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
    if _HAS_NUMBA and not surface_tests and hasattr(field, "x_m") and hasattr(field, "y_m") and hasattr(field, "z_m"):
        numba_bounds = np.zeros((3, 2), dtype=np.float64) if bounds is None else bounds.astype(np.float64)
        return _numba_integrate_ray(
            np.asarray(r0, dtype=np.float64).reshape(3),
            np.asarray(dir0, dtype=np.float64).reshape(3),
            np.asarray(field.x_m, dtype=np.float64),
            np.asarray(field.y_m, dtype=np.float64),
            np.asarray(field.z_m, dtype=np.float64),
            np.asarray(field.n_grid, dtype=np.float64),
            np.asarray(field.grad_x, dtype=np.float64),
            np.asarray(field.grad_y, dtype=np.float64),
            np.asarray(field.grad_z, dtype=np.float64),
            float(ds),
            int(steps),
            bool(adaptive),
            float(tol),
            float(ds / 32.0 if min_step is None else min_step),
            float(ds * 5.0 if max_step is None else max_step),
            numba_bounds,
            bounds is not None,
            bool(stop_on_exit),
            bool(has_ground),
            np.zeros(3, dtype=np.float64) if not has_ground else ground_origin.astype(np.float64),
            np.zeros(3, dtype=np.float64) if not has_ground else ground_e.astype(np.float64),
            np.zeros(3, dtype=np.float64) if not has_ground else ground_n.astype(np.float64),
            np.zeros(3, dtype=np.float64) if not has_ground else ground_u.astype(np.float64),
            float(ground_a_m * ground_a_m),
            float(ground_b_m * ground_b_m),
        )

    def should_stop(pos):
        if bounds is not None and stop_on_exit:
            if np.any(pos < bounds[:, 0]) or np.any(pos > bounds[:, 1]):
                return True
        if has_ground and _ground_hit_ellipsoid(pos, ground_origin, ground_e, ground_n, ground_u, ground_a_m, ground_b_m):
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
