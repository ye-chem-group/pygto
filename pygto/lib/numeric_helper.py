import sys
import numpy as np


def softplus(z):
    ''' Return np.log(1. + np.exp(z)). The use of `np.logaddexp` is for numerical stability.
    '''
    return np.logaddexp(0.0, z)


def soft_clip(x, xmin, xmax, s=10.0):
    ''' Soft clip input x so that xmin <= soft_clip(x) <= xmax.
    '''
    x = np.asarray(x)
    return (
        xmin
        + softplus(s * (x - xmin)) / s
        - softplus(s * (x - xmax)) / s
    )


def soft_log_clip(x, amin, amax, s=10.0):
    ''' Soft clip input x in log space so that amin <= soft_log_clip(x) <= amax.
    '''
    x = np.asarray(x)

    logamin = np.log(amin)
    logamax = np.log(amax)

    loga = soft_clip(x, logamin, logamax, s)
    return np.exp(loga)


def inverse_soft_clip(y, xmin, xmax, s=10.0, margin=None):
    ''' For a given y = soft_clip(x, xmin, xmax, s), find x.

        Safe by default: if y is outside [xmin, xmax], or too close to the
        boundary, it is clipped to the interior before inversion.
    '''
    y_is_scalar = np.isscalar(y)
    y = np.asarray(y, dtype=float)

    if margin is None:
        margin = 1e-3 * (xmax - xmin)

    if margin <= 0:
        raise ValueError("margin must be positive.")

    if xmin + margin >= xmax - margin:
        raise ValueError("margin is too large compared with xmax - xmin.")

    # Clip to the open interior
    y_safe = np.clip(y, xmin + margin, xmax - margin)

    z_num = s * (y_safe - xmin)
    z_den = s * (y_safe - xmax)

    log_num = np.log(np.expm1(z_num))
    log_den = np.log1p(-np.exp(z_den))

    x = xmin + (log_num - log_den) / s

    if y_is_scalar:
        return x.item()
    return x


def inverse_soft_log_clip(y, amin, amax, s=10.0, margin=None):
    ''' For a given y = soft_log_clip(x, amin, amax, s), find x.

        Safe by default: y is clipped to the interior in log space before inversion.
    '''
    if amin <= 0 or amax <= 0:
        raise ValueError("amin and amax must be positive for log-space clipping.")

    if amin >= amax:
        raise ValueError("amin must be smaller than amax.")

    y_is_scalar = np.isscalar(y)
    y = np.asarray(y, dtype=float)

    logamin = np.log(amin)
    logamax = np.log(amax)

    if margin is None:
        margin = 1e-3 * (logamax - logamin)

    if margin <= 0:
        raise ValueError("margin must be positive.")

    if logamin + margin >= logamax - margin:
        raise ValueError("margin is too large compared with log(amax) - log(amin).")

    # First make y positive, then clip in log space.
    # Nonpositive y cannot be logged, so fall back to amin before taking log.
    y_positive = np.maximum(y, amin)
    logy = np.log(y_positive)
    logy_safe = np.clip(logy, logamin + margin, logamax - margin)

    x = inverse_soft_clip(
        logy_safe,
        logamin,
        logamax,
        s=s,
        margin=margin,
    )

    return x


class FloatSum:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            try:
                val = float(val)
            except (TypeError, ValueError):
                raise TypeError(f"{key} must be convertible to float, got {type(val).__name__}")
            setattr(self, key, val)

    @property
    def value(self):
        return sum(self.__dict__.values())

    def __float__(self):
        return self.value

    def __sub__(self, other):
        return self.value - other

    def __rsub__(self, other):
        return other - self.value

    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return other + self.value

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return other * self.value

    def __truediv__(self, other):
        return self.value / other

    def __rtruediv__(self, other):
        return other / self.value

    def __repr__(self):
        return f"{type(self).__name__}({self.__dict__}, value={self.value})"


def filter_by_range(a, amin=None, amax=None):
    ''' Discard elements in a outside [amin, amax].
    '''
    arr = np.asarray(a)
    mask = np.ones(arr.shape, dtype=bool)

    if amin is not None:
        mask &= arr >= amin
    if amax is not None:
        mask &= arr <= amax

    out = arr[mask]

    if np.isscalar(a):
        if out.size == 0:
            return None
        return out.item()

    return out


if __name__ == '__main__':
    # linear clip
    amin = 1
    amax = 5
    k = 10.

    for a in [-10, 3, 10]:
        a1 = soft_clip(a, amin, amax, k)
        print(f'a= {a: 5.1f}  a1= {a1: 5.1f}')

    for a1 in [-10, 3, 10]:
        a = inverse_soft_clip(a1, amin, amax, k)
        a2 = soft_clip(a, amin, amax, k)
        print(f'a1= {a1: 5.1f}  a2= {a2: 5.1f}')

    # log-scale clip
    amin = 1e-3
    amax = 1e3
    k = 10.

    for a in [1e-5, 1e-1, 1e5]:
        x = np.log(a)
        a1 = soft_log_clip(x, amin, amax, k)
        print(f'a= {a: .3e}  a1= {a1: .3e}')

    for a1 in [1e-5, 1e-1, 1e5]:
        a = inverse_soft_log_clip(a1, amin, amax, k)
        x = np.log(a)
        a2 = soft_log_clip(x, amin, amax, k)
        print(f'a1= {a1: .3e}  a2= {a2: .3e}')
