import numpy as np

from pygto import lib
from pygto.basis import BasisSpec, Channel


lnames = 'spdfghikl'
LMAX = len(lnames)-1


class ContractedBasis(lib.StreamObject):

    _keys = {
        'atm',
    }

    def __init__(self, channels):

        # attribute from initialization
        self.channels = channels

        # attribute with default
        self.atm = None

    @classmethod
    def init_from_pyscf_basis(cls, basis, atm=None, emin=None, emax=None, keep_l=None):
        basis_data = parse_pyscf_basis(basis)
        return cls.init_from_basis_data(basis_data, atm, emin, emax, keep_l)

    @classmethod
    def init_from_basis_data(cls, basis_data, atm=None, emin=None, emax=None, keep_l=None):
        basis_data = filter_basis_data_by_l(basis_data, keep_l)
        basis_data = filter_basis_data_by_range(basis_data, emin, emax)
        channels = [ContractedChannel(l, ecs[:,0], ecs[:,1:]) for l,ecs in basis_data]
        return cls(channels).set(atm=atm)

    @property
    def nprim(self):
        ''' Return total number of primitives (not counting m components)
        '''
        return sum([c.nprim for c in self.channels])

    @property
    def nctr(self):
        ''' Return total number of contracted functions (not counting m components)
        '''
        return sum([c.nctr for c in self.channels])

    @property
    def nao(self):
        ''' Return total number of basis functions (counting m components)
        '''
        return sum([c.nao for c in self.channels])

    @property
    def nbas(self):
        ''' Return total number of basis functions (NOT counting m components)
        '''
        return sum([c.nbas for c in self.channels])

    @property
    def nchannel(self):
        ''' Return total number of channels
        '''
        return len(self.channels)

    @property
    def structure_primitive(self):
        data = [(c.l, c.nprim) for c in self.channels]
        nprim_by_l = np.zeros(LMAX+1, dtype=int)
        for l in range(LMAX+1):
            for dat in data:
                if dat[0] == l:
                    nprim_by_l[l] += dat[1]
        return ','.join([f'{nprim}{lnames[l]}' for l,nprim in enumerate(nprim_by_l) if nprim > 0])

    @property
    def structure_contracted(self):
        data = [(c.l, c.nctr) for c in self.channels]
        nctr_by_l = np.zeros(LMAX+1, dtype=int)
        for l in range(LMAX+1):
            for dat in data:
                if dat[0] == l:
                    nctr_by_l[l] += dat[1]
        return ','.join([f'{nctr}{lnames[l]}' for l,nctr in enumerate(nctr_by_l) if nctr > 0])

    @property
    def structure(self):
        return self.structure_contracted

    @property
    def contraction_summary(self):
        return '(%s) -> [%s]' % (self.structure_primitive, self.structure_contracted)

    @property
    def angular_momenta(self):
        ''' Return the angular momentum of each channel
        '''
        return [c.l for c in self.channels]

    @property
    def pyscf_basis(self):
        return self.get_pyscf_basis()

    def channel_nprim(self, channel_idx):
        self._check_channel_idx(channel_idx)
        return self.channels[channel_idx].nprim

    def channel_nctr(self, channel_idx):
        self._check_channel_idx(channel_idx)
        return self.channels[channel_idx].nctr

    def channel_l(self, channel_idx):
        self._check_channel_idx(channel_idx)
        return self.channels[channel_idx].l

    def _check_channel_idx(self, channel_idx):
        if channel_idx < 0 or channel_idx >= self.nchannel:
            raise IndexError('channel_idx out of range (0 ≤ channel_idx ≤ %d)'%(self.nchannel-1))

    def copy(self):
        new = self.__class__([c.copy() for c in self.channels])
        for k in self._keys:
            setattr(new, k, getattr(self, k))
        return new

    get_pyscf_basis = BasisSpec.get_pyscf_basis
    get_basis_str_nwchem = BasisSpec.get_basis_str_nwchem
    dump_basis_nwchem = BasisSpec.dump_basis_nwchem
    dump_basis = BasisSpec.dump_basis
    dump_channel_basis = BasisSpec.dump_channel_basis


class ContractedChannel(lib.StreamObject):

    def __init__(self, l, exponents, coefficients):
        self.l = l
        self._set_data(exponents, coefficients)

    def _set_data(self, exponents, coefficients):
        exponents = np.asarray(exponents, dtype=float)
        coefficients = np.asarray(coefficients, dtype=float)

        if exponents.ndim != 1:
            raise ValueError('Exponents must be a one-dimensional array.')
        if coefficients.ndim == 1:
            coefficients = coefficients.reshape(-1,1)
        elif coefficients.ndim != 2:
            raise ValueError('Coefficients must be a one- or two-dimensional array.')
        if len(exponents) != coefficients.shape[0]:
            raise ValueError('Exponents and coefficients must have the same nprim.')
        if len(exponents) == 0:
            raise ValueError('At least one primitive is required.')
        if coefficients.shape[1] == 0:
            raise ValueError('At least one contraction is required.')
        if np.any(~np.isfinite(exponents)) or np.any(exponents <= 0.):
            raise ValueError('Exponents must be finite and strictly positive.')
        if np.any(~np.isfinite(coefficients)):
            raise ValueError('Coefficients must be finite.')

        order = np.argsort(exponents)
        self._exponents = exponents[order].copy()
        self._coefficients = coefficients[order].copy()

    def set_data(self, exponents, coefficients):
        self._set_data(exponents, coefficients)
        return self

    def with_data(self, exponents, coefficients):
        return self.__class__(self.l, exponents, coefficients)

    @property
    def ecs(self):
        return np.column_stack((self._exponents, self._coefficients))

    @ecs.setter
    def ecs(self, value):
        ecs = np.asarray(value, dtype=float)
        if ecs.ndim == 1: ecs = ecs.reshape(1,-1)
        elif ecs.ndim != 2:
            raise ValueError('ecs must be a one- or two-dimensional array.')
        if ecs.shape[1] < 2:
            raise ValueError('ecs must contain exponents and at least one coefficient.')
        self._set_data(ecs[:,0], ecs[:,1:])

    @property
    def nprim(self):
        return self.ecs.shape[0]

    @property
    def nctr(self):
        return self.ecs.shape[1]-1

    @property
    def nbas(self):
        return self.nctr

    @property
    def nao(self):
        dgen = self.l * 2 + 1
        return self.nbas * dgen

    @property
    def exponents(self):
        ''' Return exponents in ascending order (to match Channel.exponents)
        '''
        return self._exponents.copy()

    @exponents.setter
    def exponents(self, value):
        self._set_data(value, self._coefficients)

    @property
    def coefficients(self):
        ''' Return coefficients sorted by exponents in ascending order
        '''
        return self._coefficients.copy()

    @coefficients.setter
    def coefficients(self, value):
        self._set_data(self._exponents, value)

    @property
    def structure_primitive(self):
        return f'{self.nprim}{lnames[self.l]}'

    @property
    def structure_contracted(self):
        return f'{self.nctr}{lnames[self.l]}'

    @property
    def structure(self):
        return self.structure_contracted

    @property
    def contraction_summary(self):
        return '(%s) -> [%s]' % (self.structure_primitive, self.structure_contracted)

    @property
    def pyscf_basis(self):
        ''' Return PySCF basis set for this channel
        '''
        return self.get_pyscf_basis()

    def get_pyscf_basis(self, emin=None, emax=None, sort=True):
        ''' Return PySCF-format basis.

            Args:
                emin/emax (float):
                    Exponents outside [emin, emax] will be discarded. Default is None.
                sort (bool):
                    Whether to sort the exponents in *descending* order. Default is True.
        '''
        basis_data = filter_basis_data_by_range([[self.l, self.ecs]], emin, emax)
        if len(basis_data) == 0:
            return []

        if sort: basis_data = sort_basis_data_by_exponents(basis_data)
        l, ecs = basis_data[0]
        return [[l] + [(x[0], *x[1:]) for x in ecs]]

    def copy(self):
        return self.__class__(self.l, self.exponents, self.coefficients)

    get_basis_str_nwchem = Channel.get_basis_str_nwchem
    dump_basis_nwchem = Channel.dump_basis_nwchem
    dump_basis = Channel.dump_basis


def parse_pyscf_basis(basis):
    basis_data = []
    for b in basis:
        l = b[0]
        ecs = np.asarray(b[1:]).copy()
        if ecs.ndim == 1: ecs = ecs.reshape(1,-1)
        basis_data.append( (l, ecs) )
    return basis_data


def filter_basis_data_by_l(basis_data, keep_l=None):
    if keep_l is None:
        return basis_data

    keep_l = lib.to_int_list(keep_l)
    basis_data = [b for b in basis_data if b[0] in keep_l]
    return basis_data


def filter_basis_data_by_range(basis_data, emin=None, emax=None):
    if emin is None and emax is None:
        return basis_data

    if emin is None:
        get_mask = lambda x: x <= emax
    elif emax is None:
        get_mask = lambda x: x >= emin
    else:
        get_mask = lambda x: (x >= emin) & (x <= emax)

    new = []
    for l,ecs in basis_data:
        mask = get_mask(ecs[:,0])
        if np.count_nonzero(mask) == 0:
            continue
        ecs = ecs[mask]
        new.append( (l,ecs) )

    return new


def sort_basis_data_by_exponents(basis_data, descending=True):
    new = []
    for l,ecs in basis_data:
        order = np.argsort(ecs[:,0])
        if descending: order = order[::-1]
        new.append( (l,ecs[order].copy()) )
    return new


if __name__ == '__main__':
    from pyscf import gto

    atm = 'C'
    fbas = 'cc-pvdz'
    basis = gto.basis.load(fbas, atm)

    spec = ContractedBasis.init_from_pyscf_basis(basis, atm=atm)
    spec.dump_basis()
    for i in range(spec.nchannel):
        print(spec.channel_l(i), spec.channel_nprim(i), spec.channel_nctr(i))
    print(spec.structure_primitive)
    print(spec.structure_contracted)
    print(spec.contraction_summary)
