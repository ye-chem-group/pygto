import numpy as np


__all__ = ["numer_grad", "numer_hess"]


def _finite_difference_step(x, abs_step=1e-4, rel_step=1e-4):
    x = np.asarray(x, dtype=float)
    return np.maximum(abs_step, rel_step * np.maximum(np.abs(x), 1.0))


def numer_grad(func, x, delta=None, abs_step=1e-4, rel_step=1e-4):
    """Numerical gradient."""
    x = np.asarray(x, dtype=float)
    n = x.size
    g = np.zeros_like(x)
    steps = (
        np.full(n, float(delta), dtype=float)
        if delta is not None
        else _finite_difference_step(x, abs_step=abs_step, rel_step=rel_step)
    )
    for i in range(n):
        dx = np.zeros_like(x)
        dx[i] = steps[i] * 0.5
        g[i] = (func(x + dx) - func(x - dx)) / steps[i]
    return g


def numer_hess(func, x, delta=None, abs_step=1e-3, rel_step=1e-3, verbose=False):
    x = np.asarray(x, dtype=float)
    n = x.size
    h = np.zeros((n, n), dtype=x.dtype)
    steps = (
        np.full(n, float(delta), dtype=float)
        if delta is not None
        else _finite_difference_step(x, abs_step=abs_step, rel_step=rel_step)
    )
    for i in range(n):
        if verbose:
            print("Numerical Hess %d/%d" % (i + 1, n), flush=True)
        dxi = np.zeros_like(x)
        dxi[i] = steps[i] * 0.5
        for j in range(i + 1):
            dxj = np.zeros_like(x)
            dxj[j] = steps[j] * 0.5
            hij = (
                func(x + dxi + dxj)
                + func(x - dxi - dxj)
                - func(x + dxi - dxj)
                - func(x - dxi + dxj)
            ) / (steps[i] * steps[j])
            h[i, j] = h[j, i] = hij
    return h
