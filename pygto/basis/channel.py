import sys
import numpy as np

from pygto.lib import soft_clip, soft_log_clip, inverse_soft_clip, inverse_soft_log_clip
from pygto.lib import filter_by_range


REPEAT_THR = 1.01
AMIN_MIN = 1E-3
AMAX_MAX = 1E10
AMINMAX_K = 10.
BETA_MIN = 1.3
BETA_MAX = 10
BETA_K = 10.


class Channel:

    amin_min = AMIN_MIN
    amax_max = AMAX_MAX
    aminmax_k = AMINMAX_K

    def __init__(self, l, exponents):
        self.l = int(l)
        self.n = len(exponents)

        if any([e <= 0. for e in exponents]):
            raise ValueError('Exponents must be strictly positive.')

        if self.n == 0:
            self._parameters = np.asarray([], dtype=float)
        else:
            self._parameters = self.exponents_to_parameters(exponents)

    @classmethod
    def init_from_pyscf_basis(cls, l, basis, repeat_thr=REPEAT_THR, emin=None, emax=None):
        exponents = exponents_from_pyscf_basis_by_l(basis, l, repeat_thr, emin, emax)
        return cls(l, exponents)

    def convert_to(self, channel_type):
        ''' Convert a copy of current Channel into specified channel type
        '''
        ct = channel_type
        if not isinstance(ct, str):
            raise TypeError('Channel type must be str.')

        ct = ct.lower()
        if ct == 'etb':
            new = ETB(self.l, self.exponents)
        elif ct == 'full':
            new = Full(self.l, self.exponents)
        else:
            raise TypeError('Unknown channel type. Acceptable values are "etb" and "full".')

        return new

    def __repr__(self):
        return f'{self.__class__.__name__}({self.structure})'

    # parameters
    @property
    def nparam(self):
        return len(self.parameters)

    @property
    def parameters(self):
        return np.asarray(self._parameters, dtype=float).copy()

    @parameters.setter
    def parameters(self, value):
        ''' Update parameters in place
        '''
        value = np.asarray(value, dtype=float)
        if value.size != self.nparam:
            raise ValueError(
                'Expected %d parameters, got %d' % (self.nparam, value.size)
            )
        self._parameters = value.copy()

    @property
    def convergence_parameters(self):
        ''' Parameters for the optimizer to calculate ∆x to check convergence,
            chosen to be log(exponents)
        '''
        return np.log(self.exponents)

    def with_parameters(self, value):
        ''' Return a new Channel with updated parameters
        '''
        channel = self.copy()
        channel.parameters = value
        return channel

    # properties
    @property
    def nao(self):
        dgen = self.l * 2 + 1
        return self.n * dgen

    @property
    def nbas(self):
        return self.n

    @property
    def structure(self):
        lstr = 'spdfghikl'[self.l]
        return f'{self.n}{lstr}'

    @property
    def pyscf_basis(self):
        ''' Return PySCF basis set for this channel
        '''
        return self.get_pyscf_basis()

    def get_pyscf_basis(self, emin=None, emax=None):
        exponents = filter_by_range(self.exponents, emin, emax)
        return [(int(self.l), (e, 1.)) for e in exponents]

    def copy(self):
        return self.__class__(self.l, self.exponents)

    def merge(self, others, repeat_thr=REPEAT_THR):
        ''' Merge the current channel with other channels of same angular momentum.

            `others` can be either a single other channel or a list of other channels.
            A TypeError will be raised if any of the other channels have (i) different `l` or (ii) different channel type (e.g., ETB vs Full) compared to the current one.
        '''
        if isinstance(others, Channel):
            others = [others]

        if not all([isinstance(other, self.__class__) for other in others]):
            raise TypeError('Only same-type channels can be merged')

        if not all([other.l == self.l for other in others]):
            raise TypeError('Only channels of same `l` can be merged')

        exponents = np.hstack((
            [self.exponents.copy()] +
            [other.exponents.copy() for other in others]
        ))

        exponents = remove_repeated_exponents(exponents, repeat_thr)

        return self.__class__(self.l, exponents)

    def remove_one_exponent_candidates(self, upscale=1.2, downscale=0.8):
        channels = []
        for exponents in remove_one_exponent_candidates(self.exponents, upscale, downscale):
            channels.append( self.__class__(self.l, exponents) )
        return channels

    def add_one_exponent_candidates(self, upscale=1.2, downscale=0.8, emin=0.01):
        channels = []
        for exponents in add_one_exponent_candidates(self.exponents, upscale, downscale, emin):
            channels.append( self.__class__(self.l, exponents) )
        return channels

    def get_ratio_penalty(self, ratio_min=None, strength=None):
        ''' Penalty on two exponents being too close.

            Meth:
                r(i) = e(i+1)/e(i)
                penalty = strength * sum_i ( min(r(i) - rmin, 0) )^2
        '''
        if ratio_min is None or strength is None:
            return 0.

        if ratio_min < 0 or strength < 0:
            raise ValueError('ratio_min and strength must both be positive.')

        if self.nbas <= 1:
            return 0.

        es = np.sort(self.exponents)
        ratio = es[1:] / es[:-1]
        gap = ratio_min - ratio
        violation = gap[gap > 0]
        penalty = strength * sum(violation**2)

        return penalty

    def get_basis_str_nwchem(self, atm=None, header=True):
        ''' Return NWChem format basis string for this channel
        '''
        if atm is None: atm = 'X'
        lname = 'SPDFGHIKL'[self.l]
        s = []
        if header:
            struct = self.structure
            s.append( f'#BASIS SET: ({struct}) -> [{struct}]' )
        for e in np.sort(self.exponents)[::-1]:
            s.append( f'{atm}  {lname}' )
            s.append( f'{e:15.7f}  {1.: .6e}' )
        return '\n'.join(s)

    def dump_basis_nwchem(self, des=None, atm=None, header=True):
        ''' Print NWChem format basis string to given destination
        '''
        basis_str = self.get_basis_str_nwchem(atm=atm, header=header)
        if des is None: des = sys.stdout
        des.write(basis_str + '\n')

    get_basis_str = get_basis_str_nwchem
    dump_basis = dump_basis_nwchem


def exponents_from_pyscf_basis_by_l(basis, l, repeat_thr=REPEAT_THR, emin=None, emax=None):
    ''' Extract exponents of a given angular momentum from a PySCF basis
    '''
    exponents = []
    for b in basis:
        if l != int(b[0]): continue

        ecs = np.asarray(b[1:])
        if ecs.ndim == 1:
            es = [ecs[0]]
        else:
            es = ecs[:,0]

        exponents = np.hstack((exponents, es))

    exponents = filter_by_range(exponents, emin, emax)
    exponents = remove_repeated_exponents(exponents, repeat_thr)
    return exponents


def remove_repeated_exponents(exponents, repeat_thr=REPEAT_THR):
    ''' Sort exponents in ascending order and remove repeated ones,
        defined by ratio less than `repeat_thr`.
    '''
    if len(exponents) == 0:
        return exponents

    exponents = np.sort(exponents)
    rats = exponents[1:] / exponents[:-1]
    exponents = np.hstack(([exponents[0]], exponents[1:][np.where(rats > repeat_thr)[0]]))
    return exponents


def add_one_exponent_candidates(es, upscale=1.2, downscale=0.8, emin=0.01):
    es = np.sort(es)
    ne = es.size

    # add a high exponent
    es_high = np.zeros(ne + 1)
    es_high[:ne] = es * downscale
    es_high[ne] = es[-1] * 1.5

    # add a higher exponent
    es_higher = np.zeros(ne + 1)
    es_higher[:ne] = es * 1
    es_higher[ne] = es[-1] * 3

    # add a low exponent
    es_low = np.zeros(ne + 1)
    es_low[1:] = es * upscale
    es_low[0] = es[0] * 0.7

    # add a lower exponent
    es_lower = np.zeros(ne + 1)
    es_lower[1:] = es * 1
    es_lower[0] = max(es[0] * 0.3, emin)

    return es_higher, es_high, es_low, es_lower


def remove_one_exponent_candidates(es, upscale=1.2, downscale=0.8):
    es = np.sort(es)

    # remove a high exponent
    es_high = es[:-1] * upscale

    # remove a low exponent
    es_low = es[1:] * downscale

    return es_high, es_low


class ETB(Channel):

    beta_min = BETA_MIN
    beta_max = BETA_MAX
    beta_k = BETA_K

    def exponents_to_parameters(self, exponents):
        n, amin, beta = exponents_to_ETB(exponents)
        p0 = inverse_soft_log_clip(amin, self.amin_min, self.amax_max, self.aminmax_k)
        if n > 1:
            p1 = inverse_soft_clip(beta, self.beta_min, self.beta_max, self.beta_k)
            return np.asarray([p0,p1])
        else:
            return np.asarray([p0])

    @property
    def amin(self):
        return soft_log_clip(self._parameters[0], self.amin_min, self.amax_max, self.aminmax_k)

    @property
    def beta(self):
        if self.n > 1:
            return soft_clip(self._parameters[1], self.beta_min, self.beta_max, self.beta_k)
        else:
            return 1.

    @property
    def exponents(self):
        ''' Return a list of exponents
        '''
        if self.n == 0:
            return np.asarray([], dtype=float)
        exponents = ETB_to_exponents(self.n, self.amin, self.beta)
        return np.sort(exponents)[::-1]


def exponents_to_ETB(exponents):
    ''' Convert exponents to ETB parameters: n, amin, and beta
    '''
    exponents = np.asarray(exponents)
    n = len(exponents)
    if n <= 0:
        raise RuntimeError

    exponents = np.sort(exponents)

    if n == 1:
        amin = exponents[0]
        beta = 1
    else:
        try:
            amin, beta = _fit_ETB_lstsq(exponents)
        except Exception as e:
            print(f'Least-square fit of ETB failed with {type(e).__name__}. '
                  'Switching to min-max fit', flush=True)
            amin, beta = _fit_ETB_minmax(exponents)

    return n, amin, beta

def _fit_ETB_minmax(exponents):
    ''' Choose ETB parameters such that
            - amin = min(exponents)
            - amax = amin * beta**(n-1) = max(exponents)
    '''
    n = len(exponents)
    amin = np.min(exponents)
    amax = np.max(exponents)
    beta = np.exp(np.log(amax/amin) / (n-1))
    return amin, beta

def _fit_ETB_lstsq(exponents):
    ''' Choose ETB parameters to minimize 2-norm error against exponents.
    '''
    exponents = np.sort(exponents)
    n = len(exponents)

    rn = np.arange(n)
    a = np.asarray([
        [n, rn.sum()],
        [rn.sum(), (rn**2).sum()]
    ])
    loge = np.log(exponents)
    b = np.asarray([
        loge.sum(), (loge*rn).sum()
    ])
    x = np.linalg.solve(a, b)
    amin, beta = np.exp(x)

    return amin, beta


def ETB_to_exponents(n, amin, beta):
    ''' Generate exponents from ETB parameters. For `n = 1`, `beta` is ignored.
    '''
    if n == 1:
        return np.asarray([amin])

    return np.exp(np.log(amin) + np.arange(n)*np.log(beta))


class Full(Channel):

    def exponents_to_parameters(self, exponents):
        exponents = np.sort(exponents)
        parameters = inverse_soft_log_clip(exponents, self.amin_min, self.amax_max, self.aminmax_k)
        return parameters

    @property
    def exponents(self):
        ''' Return a list of exponents
        '''
        return soft_log_clip(self._parameters, self.amin_min, self.amax_max, self.aminmax_k)


if __name__ == '__main__':
    exponents = [0.14, 0.56, 0.9, 2.7, 6.35, 19.2]

    l = 0
    channel = ETB(l, exponents)
    print(channel)
    print(channel.exponents)
    print(channel.amin, channel.beta)
    print(channel.nao)
    print(channel.nbas)
    print(channel.nparam)
    # print(channel.get_basis_str())
    channel.dump_basis(atm='C')

    l = 2
    channel = Full(l, exponents)
    print(channel)
    print(channel.exponents)
    print(channel.nao)
    print(channel.nbas)
    print(channel.nparam)
    # print(channel.get_basis_str())
    channel.dump_basis(atm='C')
