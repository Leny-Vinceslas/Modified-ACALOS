import numpy as np

def evaluate_bh2002_curve(x, fitparams, inverse=False):
    """
    Loudness mapping using a piecewise linear + quadratic Bezier transition.

    Args:
        x: array-like; levels (forward) or CU values (inverse).
        fitparams: [Lcut, m_low, m_high], slopes in CU/dB.
        inverse: False for level->CU, True for CU->level.

    Returns:
        np.ndarray of mapped values (same shape as input).
    """
    x = _to_1d_array(x)
    if len(fitparams) != 3:
        raise ValueError("fitparams must be [Lcut, m_low, m_high]")

    Lcut, m_low, m_high = fitparams
    m_low = _clip_positive(m_low)
    m_high = _clip_positive(m_high)

    # Endpoints for CU=15 and CU=35
    L15 = _linear_y_to_x(15.0, Lcut, 25.0, m_low)
    L35 = _linear_y_to_x(35.0, Lcut, 25.0, m_high)

    # Control point chosen to match slopes at both ends (C1 continuity)
    denom = (m_low - m_high) if not np.isclose(m_low, m_high) else 1e-9
    x1 = (20.0 - m_high * L35 + m_low * L15) / denom
    x1 = np.clip(x1, min(L15, L35) + 1e-6, max(L15, L35) - 1e-6)
    y1 = 15.0 + m_low * (x1 - L15)

    control = np.array([[L15, x1, L35], [15.0, y1, 35.0]])

    if not inverse:
        return _forward_level_to_cu_curve(x, Lcut, m_low, m_high, control)
    return _inverse_cu_curve(x, Lcut, m_low, m_high, control)


# ---------- helpers ----------

def _to_1d_array(x):
    if isinstance(x, np.ndarray):
        return x.flatten()
    return np.array(x, dtype=float).flatten()

def _clip_positive(val, floor=1e-3):
    if val <= 0:
        return max(floor, abs(val))
    return val

def _linear_x_to_y(x, x0, y0, slope):
    return y0 + slope * (x - x0)

def _linear_y_to_x(y, x0, y0, slope):
    return (y - y0) / slope + x0

def _bezier_t_for_x(x, control):
    """
    Solve x(t) = (1-t)^2*P0 + 2(1-t)t*P1 + t^2*P2 for t in [0,1].
    Returns (t_values, fail_flag).
    """
    x0, x1, x2 = control[0]
    a = x0 - 2 * x1 + x2
    b = 2 * (x1 - x0)
    c = x0 - x
    if np.isclose(a, 0):
        t = -c / b
        return np.clip(t, 0.0, 1.0), False

    disc = b * b - 4 * a * c
    if np.any(disc < 0):
        return np.zeros_like(x), True

    sqrt_disc = np.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)

    # pick root closest to linear guess
    t_lin = np.clip((x - x0) / (x2 - x0 + 1e-12), 0.0, 1.0)
    candidates = np.stack([t1, t2], axis=-1)
    valid = (candidates >= 0) & (candidates <= 1)
    dist = np.abs(candidates - t_lin[..., None])
    dist[~valid] = np.inf
    t = candidates[np.arange(len(t1)), np.argmin(dist, axis=-1)]
    fail = np.any(~valid.any(axis=-1)) | np.isnan(t).any()
    return t, fail

def _evaluate_quadratic_bezier(control, t):
    P0, P1, P2 = control.T
    return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2

def _forward_level_to_cu_curve(levels, Lcut, m_low, m_high, control):
    cu = np.empty_like(levels, dtype=float)

    below = levels <= control[0, 0]
    above = levels >= control[0, 2]
    mid = ~(below | above)

    cu[below] = _linear_x_to_y(levels[below], Lcut, 25.0, m_low)
    cu[above] = _linear_x_to_y(levels[above], Lcut, 25.0, m_high)

    if np.any(mid):
        t, fail = _bezier_t_for_x(levels[mid], control)
        if fail:
            raise RuntimeError("Bezier inversion failed in forward mapping")
        cu[mid] = _evaluate_quadratic_bezier(control[1], t)

    return np.clip(cu, 0.0, 50.0)

def _inverse_cu_curve(cu, Lcut, m_low, m_high, control):
    cu_clipped = np.clip(cu, 0.0, 50.0)
    levels = np.empty_like(cu_clipped, dtype=float)

    below = cu_clipped <= control[1, 0]
    above = cu_clipped >= control[1, 2]
    mid = ~(below | above)

    levels[below] = _linear_y_to_x(cu_clipped[below], Lcut, 25.0, m_low)
    levels[above] = _linear_y_to_x(cu_clipped[above], Lcut, 25.0, m_high)

    if np.any(mid):
        y0, y1, y2 = control[1]
        a = y0 - 2 * y1 + y2
        b = 2 * (y1 - y0)
        c = y0 - cu_clipped[mid]
        if np.isclose(a, 0).all():
            t = -c / b
        else:
            disc = b * b - 4 * a * c
            disc[disc < 0] = 0
            sqrt_disc = np.sqrt(disc)
            t1 = (-b + sqrt_disc) / (2 * a)
            t2 = (-b - sqrt_disc) / (2 * a)
            candidates = np.stack([t1, t2], axis=-1)
            valid = (candidates >= 0) & (candidates <= 1)
            t = np.where(valid.any(axis=-1), candidates[..., 0], np.nan)
            t = np.where(valid[..., 1], candidates[..., 1], t)
            if np.isnan(t).any():
                raise RuntimeError("Bezier inversion failed in inverse mapping")
        levels[mid] = _evaluate_quadratic_bezier(control[0], t)

    return levels
