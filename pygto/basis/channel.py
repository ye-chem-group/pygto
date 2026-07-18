import sys
import numpy as np

from pygto import lib


REPEAT_THR = 1.01
AMIN_MIN = 1E-3
AMAX_MAX = 1E10
AMINMAX_K = 10.
BETA_MIN = 1.3
BETA_MAX = 10
BETA_K = 10.


class Channel(lib.StreamObject):
    ''' Base class for an optimizable uncontracted angular-momentum channel.

        Args:
            l (int):
                Angular momentum.
            exponents (array_like):
                Primitive exponents.

        Attributes:
            amin_min (float):
                Lower exponent bound used by parameter transformations. Default is
                `AMIN_MIN`.
            amax_max (float):
                Upper exponent bound used by parameter transformations. Default is
                `AMAX_MAX`.
            aminmax_k (float):
                Soft-clipping strength for exponent bounds. Default is `AMINMAX_K`.
    '''

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
        ''' Initialize a channel from a PySCF-format basis.

            Args:
                l (int):
                    Angular momentum to extract.
                basis (list):
                    Basis data in PySCF format.
                repeat_thr (float):
                    Exponents with an adjacent ratio no greater than this threshold
                    are treated as repeated. Default is `REPEAT_THR`.
                emin/emax (float):
                    Exponents outside [emin, emax] are discarded. Default is None,
                    which does not impose the corresponding bound.

            Return:
                channel (Channel):
                    Channel initialized from the selected exponents.
        '''
        exponents = exponents_from_pyscf_basis_by_l(basis, l, repeat_thr, emin, emax)
        return cls(l, exponents)

    def convert_to(self, channel_type):
        ''' Convert a copy of the channel to a specified channel type.

            Args:
                channel_type (str):
                    Target channel type. Accepted values are "etb" and "full"
                    (case insensitive).

            Return:
                channel (Channel):
                    Converted copy of the channel.
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
        ''' Return a concise representation of the channel. '''
        return f'{self.__class__.__name__}({self.structure})'

    # parameters
    @property
    def nparam(self):
        ''' Return the number of optimization parameters.

            Return:
                nparam (int):
                    Number of channel parameters.
        '''
        return len(self.parameters)

    @property
    def parameters(self):
        ''' Return optimization parameters as a copied array.

            Return:
                parameters (ndarray):
                    Channel optimization parameters.
        '''
        return np.asarray(self._parameters, dtype=float).copy()

    @parameters.setter
    def parameters(self, value):
        ''' Update optimization parameters in place.

            Args:
                value (array_like):
                    New parameters. The size must equal `nparam`.
        '''
        value = np.asarray(value, dtype=float)
        if value.size != self.nparam:
            raise ValueError(
                'Expected %d parameters, got %d' % (self.nparam, value.size)
            )
        self._parameters = value.copy()

    @property
    def convergence_parameters(self):
        ''' Return parameters used by optimizers to check convergence.

            Return:
                parameters (ndarray):
                    Natural logarithms of the channel exponents.
        '''
        return np.log(self.exponents)

    def with_parameters(self, value):
        ''' Return a copy with updated optimization parameters.

            Args:
                value (array_like):
                    New channel parameters.

            Return:
                channel (Channel):
                    Copy containing the updated parameters.
        '''
        channel = self.copy()
        channel.parameters = value
        return channel

    # properties
    @property
    def exponents(self):
        ''' Return primitive exponents.

            Return:
                exponents (ndarray):
                    Primitive exponents in ascending order.

            Note:
                Subclasses must implement this property.
        '''
        raise NotImplementedError

    @exponents.setter
    def exponents(self, value):
        ''' Reset channel parameters from primitive exponents.

            Args:
                value (array_like):
                    Strictly positive primitive exponents.
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
        ''' Return the number of basis functions including m components.

            Return:
                nao (int):
                    Number of spherical atomic orbitals.
        '''
        dgen = self.l * 2 + 1
        return self.nbas * dgen

    @property
    def nbas(self):
        ''' Return the number of radial basis functions.

            Return:
                nbas (int):
                    Number of primitive basis functions excluding m components.
        '''
        return self.nexponent

    @property
    def nexponent(self):
        ''' Return the number of primitive exponents.

            Return:
                nexponent (int):
                    Number of exponents in the channel.
        '''
        return self._nexponent

    @property
    def structure(self):
        ''' Return the channel structure.

            Return:
                structure (str):
                    Basis count followed by the angular-momentum label.
        '''
        lstr = 'spdfghikl'[self.l]
        return f'{self.nbas}{lstr}'

    @property
    def pyscf_basis(self):
        ''' Return the channel in PySCF format.

            Return:
                basis (list):
                    Basis data in PySCF format.
        '''
        return self.get_pyscf_basis()

    def get_pyscf_basis(self, emin=None, emax=None, sort=True):
        ''' Return PySCF-format basis.

            Args:
                emin/emax (float):
                    Exponents outside [emin, emax] will be discarded. Default is None.
                sort (bool):
                    Whether to sort the exponents in *descending* order. Default is True.

            Return:
                basis (list):
                    Uncontracted basis data in PySCF format.
        '''
        exponents = lib.filter_by_range(self.exponents, emin, emax)
        if sort: exponents = np.sort(exponents)[::-1]
        return [(int(self.l), (e, 1.)) for e in exponents]

    def copy(self):
        ''' Return an independent copy of the channel.

            Return:
                channel (Channel):
                    Copy of the channel.
        '''
        new = self.__class__(self.l, self.exponents)
        for k in self._keys:
            setattr(new, k, getattr(self, k))
        return new

    def replace_exponents(self, exponents):
        ''' Return a copy with replaced exponents.

            Args:
                exponents (array_like):
                    Replacement primitive exponents.

            Return:
                channel (Channel):
                    Copy containing the replacement exponents.
        '''
        new = self.copy()
        new.exponents = exponents
        return new

    def replace_exponents_(self, exponents):
        ''' Replace primitive exponents in place.

            Args:
                exponents (array_like):
                    Replacement primitive exponents.
        '''
        self.exponents = exponents

    def merge(self, others, repeat_thr=REPEAT_THR):
        ''' Merge the current channel with other channels of same angular momentum.

            Args:
                others (Channel or list of Channel):
                    Same-type channels with the same angular momentum.
                repeat_thr (float):
                    Exponents with an adjacent ratio no greater than this threshold
                    are treated as repeated. Default is `REPEAT_THR`.

            Return:
                channel (Channel):
                    Copy containing the merged exponents.
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
        ''' Return candidate channels with one exponent removed.

            Args:
                upscale (float):
                    Scaling factor applied after removing the tightest exponent.
                    Default is 1.2.
                downscale (float):
                    Scaling factor applied after removing the most diffuse exponent.
                    Default is 0.8.

            Return:
                channels (list of Channel):
                    Candidate channels.
        '''
        channels = []
        for exponents in remove_one_exponent_candidates(self.exponents, upscale, downscale):
            channels.append( self.replace_exponents(exponents) )
        return channels

    def add_one_exponent_candidates(self, upscale=1.2, downscale=0.8, emin=0.01):
        ''' Return candidate channels with one exponent added.

            Args:
                upscale (float):
                    Scaling factor applied to existing exponents when adding a
                    diffuse exponent. Default is 1.2.
                downscale (float):
                    Scaling factor applied to existing exponents when adding a tight
                    exponent. Default is 0.8.
                emin (float):
                    Lower bound for the most diffuse candidate. Default is 0.01.

            Return:
                channels (list of Channel):
                    Candidate channels.
        '''
        channels = []
        for exponents in add_one_exponent_candidates(self.exponents, upscale, downscale, emin):
            channels.append( self.replace_exponents(exponents) )
        return channels

    def remove_one_exponent_candidates_rigid(self, emin=None, emax=None):
        ''' Return candidates formed by removing each selected exponent once.

            Args:
                emin/emax (float):
                    The channel is first restricted to exponents within [emin, emax].
                    Default is None, which does not impose the corresponding bound.

            Return:
                channels (list of Channel):
                    Candidate channels with one exponent removed without rescaling.
        '''
        exponents = np.sort(lib.filter_by_range(self.exponents, emin, emax))
        return [
            self.replace_exponents(subexponents)
            for subexponents in [np.delete(exponents, i) for i in range(len(exponents))]
        ]

    def filter_by_exponent_range_(self, emin=None, emax=None):
        ''' Filter primitive exponents by range in place.

            Args:
                emin/emax (float):
                    Exponents outside [emin, emax] are discarded. Default is None,
                    which does not impose the corresponding bound.
        '''
        self.exponents = lib.filter_by_range(self.exponents, emin, emax)

    def filter_by_exponent_range(self, emin=None, emax=None):
        ''' Return a copy filtered by exponent range.

            Args:
                emin/emax (float):
                    Exponents outside [emin, emax] are discarded. Default is None,
                    which does not impose the corresponding bound.

            Return:
                channel (Channel):
                    Filtered copy of the channel.
        '''
        new = self.copy()
        new.filter_by_exponent_range_(emin, emax)
        return new

    def filter_by_index_(self, exponent_idx):
        ''' Filter primitive exponents by index in place.

            Args:
                exponent_idx (int or list of int):
                    Exponent indices to retain.
        '''
        index = lib.to_int_list(exponent_idx)
        if not set(index).issubset(set(range(self.nexponent))):
            raise IndexError('exponent_idx contains index out of range.')
        self.exponents = self.exponents[index]

    def filter_by_index(self, exponent_idx):
        ''' Return a copy filtered by exponent index.

            Args:
                exponent_idx (int or list of int):
                    Exponent indices to retain.

            Return:
                channel (Channel):
                    Filtered copy of the channel.
        '''
        new = self.copy()
        new.filter_by_index_(exponent_idx)
        return new

    def get_ratio_penalty(self, ratio_min=None, strength=None):
        ''' Return a smooth penalty for adjacent exponents that are too close.

            Args:
                ratio_min (float):
                    Minimum desired ratio between adjacent exponents. Default is None,
                    which disables the penalty.
                strength (float):
                    Penalty strength. Default is None, which disables the penalty.

            Return:
                penalty (float):
                    Smooth quadratic penalty based on adjacent exponent ratios.
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
        smooth_violation = width * lib.softplus(violation / width)
        penalty = strength * sum(smooth_violation**2)

        return penalty

    def get_basis_str_nwchem(self, atm=None, header=True, sort=True):
        ''' Return the channel as an NWChem-format string.

            Args:
                atm (str):
                    Atomic symbol printed in the basis. Default is None, which uses "X".
                header (bool):
                    Whether to include the basis header. Default is True.
                sort (bool):
                    Whether to sort exponents in descending order. Default is True.

            Return:
                basis_str (str):
                    Channel data in NWChem format.
        '''
        return lib.get_basis_str_nwchem(
            self.get_pyscf_basis(sort=sort), atm, header, sort
        )

    def dump_basis_nwchem(self, stdout=None, atm=None, header=True, sort=True):
        ''' Write the channel in NWChem format.

            Args:
                stdout (file-like object):
                    Destination for the basis data. Default is None, which uses
                    `sys.stdout`.
                atm (str):
                    Atomic symbol printed in the basis. Default is None, which uses "X".
                header (bool):
                    Whether to include the basis header. Default is True.
                sort (bool):
                    Whether to sort exponents in descending order. Default is True.
        '''
        lib.dump_basis_nwchem(
            self.get_pyscf_basis(sort=sort), stdout, atm, header, sort
        )

    get_basis_str = get_basis_str_nwchem
    dump_basis = dump_basis_nwchem

    def dump_chkfile(self, chkfile, prefix=None):
        ''' Save channel data to a checkpoint file.

            Args:
                chkfile (str):
                    Path to checkpoint file.
                prefix (str):
                    Key in the checkpoint file. Default is None, which uses "channel".
        '''
        if prefix is None: prefix = 'channel'
        lib.chkfile_helper.dump(chkfile, f'{prefix}/type', self.__class__.__name__.lower())
        lib.chkfile_helper.dump(chkfile, f'{prefix}/l', self.l)
        lib.chkfile_helper.dump(chkfile, f'{prefix}/exponents', self.exponents)

    @classmethod
    def init_from_chkfile(cls, chkfile, prefix=None):
        ''' Initialize a channel from a checkpoint file.

            Args:
                chkfile (str):
                    Path to checkpoint file.
                prefix (str):
                    Key in the checkpoint file. Default is None, which uses "channel".

            Return:
                channel (Channel):
                    Channel initialized from saved data.
        '''
        if prefix is None: prefix = 'channel'
        l = int(lib.chkfile_helper.load(chkfile, f'{prefix}/l'))
        exponents = np.asarray(lib.chkfile_helper.load(chkfile, f'{prefix}/exponents'), dtype=float)
        return cls(l, exponents)


def exponents_from_pyscf_basis_by_l(basis, l, repeat_thr=REPEAT_THR, emin=None, emax=None):
    ''' Extract exponents of one angular momentum from a PySCF-format basis.

        Args:
            basis (list):
                Basis data in PySCF format.
            l (int):
                Angular momentum to extract.
            repeat_thr (float):
                Exponents with an adjacent ratio no greater than this threshold are
                treated as repeated. Default is `REPEAT_THR`.
            emin/emax (float):
                Exponents outside [emin, emax] are discarded. Default is None,
                which does not impose the corresponding bound.

        Return:
            exponents (ndarray):
                Unique exponents in ascending order.
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

    exponents = lib.filter_by_range(exponents, emin, emax)
    exponents = remove_repeated_exponents(exponents, repeat_thr)
    return exponents


def remove_repeated_exponents(exponents, repeat_thr=REPEAT_THR):
    ''' Sort exponents and remove adjacent values that are too close.

        Args:
            exponents (array_like):
                Primitive exponents.
            repeat_thr (float):
                Adjacent exponents are retained only when their ratio is greater
                than this threshold. Default is `REPEAT_THR`.

        Return:
            exponents (ndarray):
                Filtered exponents in ascending order.
    '''
    if len(exponents) == 0:
        return exponents

    exponents = np.sort(exponents)
    rats = exponents[1:] / exponents[:-1]
    exponents = np.hstack(([exponents[0]], exponents[1:][np.where(rats > repeat_thr)[0]]))
    return exponents


def add_one_exponent_candidates(es, upscale=1.2, downscale=0.8, emin=0.01):
    ''' Generate exponent arrays containing one additional exponent.

        Args:
            es (array_like):
                Primitive exponents.
            upscale (float):
                Scaling factor applied to existing exponents when adding a diffuse
                exponent. Default is 1.2.
            downscale (float):
                Scaling factor applied to existing exponents when adding a tight
                exponent. Default is 0.8.
            emin (float):
                Lower bound for the most diffuse candidate. Default is 0.01.

        Return:
            candidates (tuple of array_like):
                Four candidates ordered from tighter to more diffuse extensions.
    '''
    es = np.sort(es)
    ne = es.size

    # handle empty channel
    if ne == 0:
        elow = max(emin, 0.05)
        return [5.], [1.], [elow*3.], [elow]

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
    ''' Generate exponent arrays containing one fewer exponent.

        Args:
            es (array_like):
                Primitive exponents.
            upscale (float):
                Scaling factor applied after removing the tightest exponent.
                Default is 1.2.
            downscale (float):
                Scaling factor applied after removing the most diffuse exponent.
                Default is 0.8.

        Return:
            candidates (tuple or list of ndarray):
                Candidate exponent arrays. Empty and one-exponent inputs produce
                zero and one candidates, respectively.
    '''
    es = np.sort(es)

    # handle one-exponent and empty channels
    if len(es) <= 1:
        if len(es) == 1:
            return [np.asarray([], dtype=es.dtype)]
        else:
            return []   # no candidates

    # remove a high exponent
    es_high = es[:-1] * upscale

    # remove a low exponent
    es_low = es[1:] * downscale

    return es_high, es_low


class ETB(Channel):
    ''' An even-tempered optimizable channel.

        Exponents are represented as `amin * beta**i` for consecutive integer `i`.

        Args:
            l (int):
                Angular momentum.
            exponents (array_like):
                Primitive exponents.

        Attributes:
            beta_min (float):
                Lower bound for the geometric ratio. Default is `BETA_MIN`.
            beta_max (float):
                Upper bound for the geometric ratio. Default is `BETA_MAX`.
            beta_k (float):
                Soft-clipping strength for the geometric-ratio bounds. Default is
                `BETA_K`.
    '''

    beta_min = BETA_MIN
    beta_max = BETA_MAX
    beta_k = BETA_K

    @classmethod
    def init_from_etb_params(cls, l, nprim, amin, beta):
        ''' Initialize an ETB channel from even-tempered parameters.

            Args:
                l (int):
                    Angular momentum.
                nprim (int):
                    Number of primitive exponents.
                amin (float):
                    Smallest exponent.
                beta (float):
                    Geometric ratio between adjacent exponents.

            Return:
                channel (ETB):
                    Even-tempered channel generated from the supplied parameters.

            Note:
                `amin` and `beta` are subject to the ETB parameter bounds. `beta` is
                ignored when `nprim` is 1.
        '''
        from numbers import Integral
        if not isinstance(l, Integral):
            raise TypeError('l must be an integer.')
        elif l < 0:
            raise ValueError('l must be nonnegative.')
        if not isinstance(nprim, Integral):
            raise TypeError('nprim must be an integer.')
        elif nprim < 1:
            raise ValueError('nprim must be positive.')
        exponents = ETB_to_exponents(nprim, amin, beta)
        return cls(l, exponents)

    def exponents_to_parameters(self, exponents):
        ''' Convert primitive exponents to unconstrained ETB parameters.

            Args:
                exponents (array_like):
                    Primitive exponents.

            Return:
                parameters (ndarray):
                    Unconstrained parameters representing `amin` and, when needed,
                    `beta`.
        '''
        n, amin, beta = exponents_to_ETB(exponents)
        p0 = lib.inverse_soft_log_clip(amin, self.amin_min, self.amax_max, self.aminmax_k)
        if n > 1:
            p1 = lib.inverse_soft_clip(beta, self.beta_min, self.beta_max, self.beta_k)
            return np.asarray([p0,p1])
        else:
            return np.asarray([p0])

    @property
    def amin(self):
        ''' Return the smallest even-tempered exponent.

            Return:
                amin (float):
                    Minimum exponent after parameter transformation.
        '''
        return lib.soft_log_clip(self._parameters[0], self.amin_min, self.amax_max, self.aminmax_k)

    @property
    def beta(self):
        ''' Return the even-tempered geometric ratio.

            Return:
                beta (float):
                    Geometric ratio, or 1 for a one-exponent channel.
        '''
        if self._nexponent > 1:
            return lib.soft_clip(self._parameters[1], self.beta_min, self.beta_max, self.beta_k)
        else:
            return 1.

    @property
    def exponents(self):
        ''' Return even-tempered exponents in ascending order.

            Return:
                exponents (ndarray):
                    Primitive exponents.
        '''
        if self._nexponent == 0:
            return np.asarray([], dtype=float)
        exponents = ETB_to_exponents(self._nexponent, self.amin, self.beta)
        return np.sort(exponents)

    @exponents.setter
    def exponents(self, value):
        ''' Reset ETB parameters from primitive exponents.

            Args:
                value (array_like):
                    Strictly positive primitive exponents.
        '''
        Channel.exponents.fset(self, value)


def exponents_to_ETB(exponents):
    ''' Fit even-tempered parameters to primitive exponents.

        Args:
            exponents (array_like):
                Nonempty primitive exponents.

        Return:
            n (int):
                Number of exponents.
            amin (float):
                Smallest fitted exponent.
            beta (float):
                Fitted geometric ratio.
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
    ''' Fit ETB parameters to the minimum and maximum exponents.

        Args:
            exponents (array_like):
                At least two primitive exponents.

        Return:
            amin (float):
                Minimum input exponent.
            beta (float):
                Ratio satisfying `amin * beta**(n-1) = max(exponents)`.
    '''
    n = len(exponents)
    amin = np.min(exponents)
    amax = np.max(exponents)
    beta = np.exp(np.log(amax/amin) / (n-1))
    return amin, beta

def _fit_ETB_lstsq(exponents):
    ''' Fit ETB parameters by least squares in logarithmic space.

        Args:
            exponents (array_like):
                At least two primitive exponents.

        Return:
            amin (float):
                Fitted minimum exponent.
            beta (float):
                Fitted geometric ratio.
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
    ''' Generate primitive exponents from even-tempered parameters.

        Args:
            n (int):
                Number of exponents.
            amin (float):
                Smallest exponent.
            beta (float):
                Geometric ratio. It is ignored when `n` is 1.

        Return:
            exponents (ndarray):
                Even-tempered exponents in ascending order.
    '''
    if n == 1:
        return np.asarray([amin])

    return np.exp(np.log(amin) + np.arange(n)*np.log(beta))


class Full(Channel):
    ''' An optimizable channel with one parameter per primitive exponent.

        Args:
            l (int):
                Angular momentum.
            exponents (array_like):
                Primitive exponents.
    '''

    def exponents_to_parameters(self, exponents):
        ''' Convert primitive exponents to unconstrained parameters.

            Args:
                exponents (array_like):
                    Primitive exponents.

            Return:
                parameters (ndarray):
                    Unconstrained exponent parameters.
        '''
        exponents = np.sort(exponents)
        parameters = lib.inverse_soft_log_clip(exponents, self.amin_min, self.amax_max, self.aminmax_k)
        return parameters

    @property
    def exponents(self):
        ''' Return primitive exponents in ascending order.

            Return:
                exponents (ndarray):
                    Primitive exponents.
        '''
        # @@HY
        es = lib.soft_log_clip(self._parameters, self.amin_min, self.amax_max, self.aminmax_k)
        return np.sort(es)

    @exponents.setter
    def exponents(self, value):
        ''' Reset full-channel parameters from primitive exponents.

            Args:
                value (array_like):
                    Strictly positive primitive exponents.
        '''
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
    channel.dump_basis(atm='C')

    l = 2
    channel = Full(l, exponents)
    print(channel)
    print(channel.exponents)
    print(channel.nao)
    print(channel.nbas)
    print(channel.nparam)
    channel.dump_basis(atm='C')
