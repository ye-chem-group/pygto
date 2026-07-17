import sys
import numpy as np


def softplus(z):
    ''' Evaluate the numerically stable softplus function.

        Args:
            z (array_like):
                Input values.

        Return:
            value (scalar or ndarray):
                `log(1 + exp(z))`.
    '''
    return np.logaddexp(0.0, z)


def soft_clip(x, xmin, xmax, s=10.0):
    ''' Smoothly constrain values to an interval.

        Args:
            x (array_like):
                Unconstrained input values.
            xmin/xmax (float):
                Lower and upper bounds.
            s (float):
                Clipping sharpness. Default is 10.

        Return:
            value (scalar or ndarray):
                Soft-clipped values between `xmin` and `xmax`.
    '''
    x = np.asarray(x)
    return (
        xmin
        + softplus(s * (x - xmin)) / s
        - softplus(s * (x - xmax)) / s
    )


def soft_log_clip(x, amin, amax, s=10.0):
    ''' Map unconstrained log parameters to a positive bounded interval.

        Args:
            x (array_like):
                Unconstrained log-space parameters.
            amin/amax (float):
                Positive lower and upper bounds.
            s (float):
                Clipping sharpness. Default is 10.

        Return:
            value (scalar or ndarray):
                Positive values between `amin` and `amax`.
    '''
    x = np.asarray(x)

    logamin = np.log(amin)
    logamax = np.log(amax)

    loga = soft_clip(x, logamin, logamax, s)
    return np.exp(loga)


def inverse_soft_clip(y, xmin, xmax, s=10.0, margin=None):
    ''' Invert `soft_clip` after clipping values to a safe interior.

        Args:
            y (array_like):
                Soft-clipped values.
            xmin/xmax (float):
                Lower and upper bounds.
            s (float):
                Clipping sharpness. Default is 10.
            margin (float):
                Distance retained from each boundary. Default is None, which uses
                `1e-3 * (xmax - xmin)`.

        Return:
            x (scalar or ndarray):
                Recovered unconstrained parameters.
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
    ''' Invert `soft_log_clip` using a safe log-space interior.

        Args:
            y (array_like):
                Positive bounded values.
            amin/amax (float):
                Positive lower and upper bounds.
            s (float):
                Clipping sharpness. Default is 10.
            margin (float):
                Log-space distance retained from each boundary. Default is None,
                which uses `1e-3 * (log(amax) - log(amin))`.

        Return:
            x (scalar or ndarray):
                Recovered unconstrained log parameters.
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
    ''' A named collection of floating-point terms with scalar arithmetic.

        Args:
            kwargs (dict):
                Named values convertible to float.
    '''

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            try:
                val = float(val)
            except (TypeError, ValueError):
                raise TypeError(f"{key} must be convertible to float, got {type(val).__name__}")
            setattr(self, key, val)

    @property
    def value(self):
        ''' Return the sum of all named terms.

            Return:
                value (float):
                    Sum of stored values.
        '''
        return sum(self.__dict__.values())

    def __float__(self):
        ''' Convert the collection to its summed value. '''
        return self.value

    def __sub__(self, other):
        ''' Subtract another value from the summed value. '''
        return self.value - other

    def __rsub__(self, other):
        ''' Subtract the summed value from another value. '''
        return other - self.value

    def __add__(self, other):
        ''' Add another value to the summed value. '''
        return self.value + other

    def __radd__(self, other):
        ''' Add the summed value to another value. '''
        return other + self.value

    def __mul__(self, other):
        ''' Multiply the summed value by another value. '''
        return self.value * other

    def __rmul__(self, other):
        ''' Multiply another value by the summed value. '''
        return other * self.value

    def __truediv__(self, other):
        ''' Divide the summed value by another value. '''
        return self.value / other

    def __rtruediv__(self, other):
        ''' Divide another value by the summed value. '''
        return other / self.value

    def __repr__(self):
        ''' Return a representation containing terms and their sum. '''
        return f"{type(self).__name__}({self.__dict__}, value={self.value})"


def filter_by_range(a, amin=None, amax=None):
    ''' Discard values outside an inclusive interval.

        Args:
            a (scalar or array_like):
                Input values.
            amin/amax (float):
                Lower and upper bounds. Default is None, which does not impose the
                corresponding bound.

        Return:
            values (scalar, ndarray, or None):
                Filtered values. A scalar outside the interval returns None.
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


def to_int_list(a):
    ''' Normalize an integer or integer collection to a list.

        Args:
            a (int or iterable of int):
                Integer value or a list, tuple, set, or ndarray of integers.

        Return:
            values (list of int):
                Normalized Python integers.
    '''
    Int = (int, np.int32, np.int64)
    Iterable = (list, tuple, set, np.ndarray)

    if isinstance(a, Int):
        a = [a]
    elif isinstance(a, Iterable):
        if not all([isinstance(x, Int) for x in a]):
            raise TypeError('Some/all elements are not Integer')
        a = [int(x) for x in a]
    else:
        raise TypeError('Input must be either an Integer or a '
                        'List/Tuple/Set/NumpyArray of Integer.')

    return a


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
