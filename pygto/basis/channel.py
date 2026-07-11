import sys
import numpy as np

from pygto.lib import soft_clip, soft_log_clip, inverse_soft_clip, inverse_soft_log_clip, softplus
from pygto.lib import filter_by_range, to_int_list
from pygto.lib import StreamObject


REPEAT_THR = 1.01
AMIN_MIN = 1E-3
AMAX_MAX = 1E10
AMINMAX_K = 10.
BETA_MIN = 1.3
BETA_MAX = 10
BETA_K = 10.


class Channel(StreamObject):

    amin_min = AMIN_MIN
    amax_max = AMAX_MAX
    aminmax_k = AMINMAX_K

    _keys = {}

    def __init__(self, l, exponents):
        self.l = int(l)
        self._nexponent = None

        self.exponents = exponents

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

        for k in self._keys:
            setattr(new, k, getattr(self, k))

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
    def exponents(self):
        ''' To be implemented for each subclass
        '''
        raise NotImplementedError

    @exponents.setter
    def exponents(self, value):
        ''' Reset parameters by exponents
        '''
        value = np.asarray(value, dtype=float)
        if np.any(value <= 0.):
            raise ValueError('Exponents must be strictly positive.')

        self._nexponent = len(value)
        if self._nexponent == 0:
            self._parameters = np.asarray([], dtype=float)
        else:
            self._parameters = self.exponents_to_parameters(value)

    @property
    def nao(self):
        dgen = self.l * 2 + 1
        return self.nbas * dgen

    @property
    def nbas(self):
        return self.nexponent

    @property
    def nexponent(self):
        return self._nexponent

    @property
    def structure(self):
        lstr = 'spdfghikl'[self.l]
        return f'{self.nbas}{lstr}'

    @property
    def pyscf_basis(self):
        ''' Return PySCF basis set for this channel
        '''
        return self.get_pyscf_basis()

    def get_pyscf_basis(self, emin=None, emax=None):
        exponents = filter_by_range(self.exponents, emin, emax)
        return [(int(self.l), (e, 1.)) for e in exponents]

    def copy(self):
        new = self.__class__(self.l, self.exponents)
        for k in self._keys:
            setattr(new, k, getattr(self, k))
        return new

    def replace_exponents(self, exponents):
        ''' Return a copy with exponents replaced
        '''
        new = self.copy()
        new.exponents = exponents
        return new

    def replace_exponents_(self, exponents):
        ''' Replace exponents in place
        '''
        self.exponents = exponents

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

        return self.replace_exponents(exponents)

    def remove_one_exponent_candidates(self, upscale=1.2, downscale=0.8):
        channels = []
        for exponents in remove_one_exponent_candidates(self.exponents, upscale, downscale):
            channels.append( self.replace_exponents(exponents) )
        return channels

    def add_one_exponent_candidates(self, upscale=1.2, downscale=0.8, emin=0.01):
        channels = []
        for exponents in add_one_exponent_candidates(self.exponents, upscale, downscale, emin):
            channels.append( self.replace_exponents(exponents) )
        return channels

    def remove_one_exponent_candidates_rigid(self, emin=None, emax=None):
        exponents = np.sort(filter_by_range(self.exponents, emin, emax))
        return [
            self.replace_exponents(subexponents)
            for subexponents in [np.delete(exponents, i) for i in range(len(exponents))]
        ]

    def filter_by_exponent_range_(self, emin=None, emax=None):
        ''' Filter exponents in place by [emin, emax]
        '''
        self.exponents = filter_by_range(self.exponents, emin, emax)

    def filter_by_exponent_range(self, emin=None, emax=None):
        ''' Return a new channel where exponents are filtered by [emin, emax]
        '''
        new = self.copy()
        new.filter_by_exponent_range_(emin, emax)
        return new

    def filter_by_index_(self, exponent_idx):
        ''' Filter exponents in place by index
        '''
        index = to_int_list(exponent_idx)
        if not set(index).issubset(set(range(self.nexponent))):
            raise IndexError('exponent_idx contains index out of range.')
        self.exponents = self.exponents[index]

    def filter_by_index(self, exponent_idx):
        ''' Return a new channel where exponents are filtered by index
        '''
        new = self.copy()
        new.filter_by_index_(exponent_idx)
        return new

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
        # gap = ratio_min - ratio
        # violation = gap[gap > 0]
        # penalty = strength * sum(violation**2)

        ''' width = 1e-2 and strength = 10. rougthly leads to 1 mHa penalty when
            ratio = ratio_min-0.01. For example, ratio_min = 1.70 and ratio = 1.69.
        '''
        width = 1e-2
        violation = np.log(ratio_min) - np.log(ratio)
        smooth_violation = width * softplus(violation / width)
        penalty = strength * sum(smooth_violation**2)

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

    def dump_basis_nwchem(self, stdout=None, atm=None, header=True):
        ''' Print NWChem format basis string to given destination
        '''
        basis_str = self.get_basis_str_nwchem(atm=atm, header=header)
        if stdout is None: stdout = self.stdout
        stdout.write(basis_str + '\n')

    get_basis_str = get_basis_str_nwchem
    dump_basis = dump_basis_nwchem

    def dump_chkfile(self, chkfile, prefix=None):
        from pygto.lib import chkfile_helper
        if prefix is None: prefix = 'channel'
        chkfile_helper.dump(chkfile, f'{prefix}/type', self.__class__.__name__.lower())
        chkfile_helper.dump(chkfile, f'{prefix}/l', self.l)
        chkfile_helper.dump(chkfile, f'{prefix}/exponents', self.exponents)

    @classmethod
    def init_from_chkfile(cls, chkfile, prefix=None):
        from pygto.lib import chkfile_helper
        if prefix is None: prefix = 'channel'
        l = int(chkfile_helper.load(chkfile, f'{prefix}/l'))
        exponents = np.asarray(chkfile_helper.load(chkfile, f'{prefix}/exponents'), dtype=float)
        return cls(l, exponents)


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
        if self._nexponent > 1:
            return soft_clip(self._parameters[1], self.beta_min, self.beta_max, self.beta_k)
        else:
            return 1.

    @property
    def exponents(self):
        ''' Return a list of exponents
        '''
        if self._nexponent == 0:
            return np.asarray([], dtype=float)
        exponents = ETB_to_exponents(self._nexponent, self.amin, self.beta)
        return np.sort(exponents)[::-1]

    @exponents.setter
    def exponents(self, value):
        Channel.exponents.fset(self, value)


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

    @exponents.setter
    def exponents(self, value):
        Channel.exponents.fset(self, value)


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
