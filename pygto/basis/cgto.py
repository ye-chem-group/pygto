import numpy as np

from pygto import lib
from pygto.basis import BasisSpec, Channel


lnames = 'spdfghikl'
LMAX = len(lnames)-1


class ContractedBasis(lib.StreamObject):
    ''' A collection of channels that defines a contracted basis set.

        Args:
            channels (list of ContractedChannel):
                Contracted basis-set channels.

        Attributes:
            atm (str):
                Atomic symbol associated with the basis. Default is None.
    '''

    _keys = {
        'atm',
    }

    def __init__(self, channels):
        # attribute from initialization
        self.channels = channels

        # attribute with default
        self.atm = None

    def __repr__(self):
        ''' Return a concise representation of the ContractedBasis. '''
        return f'{self.__class__.__name__}({self.structure})'

    @classmethod
    def init_from_basis(cls, basis, atm, **kwargs):
        ''' Initialize a ContractedBasis object from common basis representations.

            Args:
                basis (str, list, or tuple):
                    Basis input. A list or tuple is interpreted as PySCF-format basis
                    data. A string may be the path to an NWChem basis file, inline
                    NWChem basis data containing a `#BASIS SET` header, or a basis-set
                    name recognized by PySCF (if installed) or Basis Set Exchange (if
                    installed or have internet access).
                atm (str):
                    Atomic symbol of the basis to load.
                kwargs (dict):
                    Additional arguments passed to `init_from_pyscf_basis`.

            Return:
                cgto (ContractedBasis):
                    ContractedBasis object initialized from `basis`.

            Note:
                String inputs are interpreted in the following order: an existing
                file, inline NWChem data, and finally a named basis set.
        '''
        if isinstance(basis, (list, tuple)):    # PySCF-format basis data
            return cls.init_from_pyscf_basis(basis, atm=atm, **kwargs)
        elif isinstance(basis, str):
            import os
            if os.path.isfile(basis):   # NWChem basis data file
                return cls.init_from_nwchem_basis(basis, atm, **kwargs)
            elif '#B' in basis:         # NWChem basis data string
                return cls.init_from_nwchem_basis(basis, atm, **kwargs)
            else:                       # Named basis
                return cls.init_from_named_basis(basis, atm, **kwargs)
        else:
            raise TypeError('basis must be a str (NWChem basis data file/string or '
                            'named basis) or a list/tuple (PySCF-format basis data).')

    @classmethod
    def init_from_nwchem_basis(cls, basis_str_or_file, atm, **kwargs):
        ''' Initialize a ContractedBasis object from an NWChem-format basis.

            Args:
                basis_str_or_file (str):
                    NWChem basis data or the path to an NWChem basis file.
                atm (str):
                    Atomic symbol of the basis to load.
                kwargs (dict):
                    Additional arguments passed to :func:`init_from_pyscf_basis`.

            Return:
                cgto (ContractedBasis):
                    ContractedBasis object initialized from the NWChem basis.
        '''
        basis = lib.load_basis_nwchem(basis_str_or_file, atm)
        return cls.init_from_pyscf_basis(basis, atm=atm, **kwargs)

    @classmethod
    def init_from_named_basis(cls, name, atm, **kwargs):
        ''' Initialize a ContractedBasis object from a named basis set.

            Args:
                name (str):
                    Basis-set name recognized by PySCF (if available) or Basis
                    Set Exchange (if available or has internet access).
                atm (str):
                    Atomic symbol of the basis to load.
                kwargs (dict):
                    Additional arguments passed to `init_from_pyscf_basis`.

            Return:
                cgto (ContractedBasis):
                    ContractedBasis object initialized from the named basis.
        '''
        if lib.has_pyscf(): # Named basis; using PySCF loader
            basis = lib.pyscf_helper.load_basis(name, atm)
        else:               # Named basis; using BasisSetExchange loader
            basis = lib.get_named_basis(name, atm)
        return cls.init_from_pyscf_basis(basis, atm=atm, **kwargs)

    @classmethod
    def init_from_pyscf_basis(cls, basis, atm=None, emin=None, emax=None, keep_l=None):
        ''' Initialize a ContractedBasis object from a PySCF-format basis.

            Args:
                basis (list):
                    Basis data in PySCF format.
                atm (str):
                    Atomic symbol associated with the basis. Default is None.
                emin/emax (float):
                    Exponents outside [emin, emax] are discarded. Default is None,
                    which does not impose the corresponding bound.
                keep_l (int or list of int):
                    Angular momenta to keep. Default is None, which keeps all channels.

            Return:
                basis (ContractedBasis):
                    Contracted basis initialized from the PySCF basis.
        '''
        basis_data = parse_pyscf_basis(basis)
        return cls.init_from_basis_data(basis_data, atm, emin, emax, keep_l)

    @classmethod
    def init_from_basis_data(cls, basis_data, atm=None, emin=None, emax=None, keep_l=None):
        ''' Initialize a ContractedBasis object from parsed basis data.

            Args:
                basis_data (list of tuple):
                    `(l, ecs)` pairs, where each row of `ecs` contains an exponent
                    followed by contraction coefficients.
                atm (str):
                    Atomic symbol associated with the basis. Default is None.
                emin/emax (float):
                    Exponents outside [emin, emax] are discarded. Default is None,
                    which does not impose the corresponding bound.
                keep_l (int or list of int):
                    Angular momenta to keep. Default is None, which keeps all channels.

            Return:
                basis (ContractedBasis):
                    Contracted basis initialized from `basis_data`.
        '''
        basis_data = filter_basis_data_by_l(basis_data, keep_l)
        basis_data = filter_basis_data_by_range(basis_data, emin, emax)
        channels = [ContractedChannel(l, ecs[:,0], ecs[:,1:]) for l,ecs in basis_data]
        return cls(channels).set(atm=atm)

    @property
    def nprim(self):
        ''' Return the number of primitives excluding m components.

            Return:
                nprim (int):
                    Total number of primitives over all channels.
        '''
        return sum([c.nprim for c in self.channels])

    @property
    def nctr(self):
        ''' Return the number of contractions excluding m components.

            Return:
                nctr (int):
                    Total number of contractions over all channels.
        '''
        return sum([c.nctr for c in self.channels])

    @property
    def nao(self):
        ''' Return the number of basis functions including m components.

            Return:
                nao (int):
                    Number of spherical atomic orbitals.
        '''
        return sum([c.nao for c in self.channels])

    @property
    def nbas(self):
        ''' Return the number of basis functions excluding m components.

            Return:
                nbas (int):
                    Number of radial contracted functions.
        '''
        return sum([c.nbas for c in self.channels])

    @property
    def nchannel(self):
        ''' Return the number of contracted channels.

            Return:
                nchannel (int):
                    Number of channels.
        '''
        return len(self.channels)

    @property
    def structure_primitive(self):
        ''' Return the primitive structure grouped by angular momentum.

            Return:
                structure (str):
                    Primitive structure, for example, "10s,5p,2d".
        '''
        data = [(c.l, c.nprim) for c in self.channels]
        nprim_by_l = np.zeros(LMAX+1, dtype=int)
        for l in range(LMAX+1):
            for dat in data:
                if dat[0] == l:
                    nprim_by_l[l] += dat[1]
        return ','.join([f'{nprim}{lnames[l]}' for l,nprim in enumerate(nprim_by_l) if nprim > 0])

    @property
    def structure_contracted(self):
        ''' Return the contracted structure grouped by angular momentum.

            Return:
                structure (str):
                    Contracted structure, for example, "4s,3p,2d".
        '''
        data = [(c.l, c.nctr) for c in self.channels]
        nctr_by_l = np.zeros(LMAX+1, dtype=int)
        for l in range(LMAX+1):
            for dat in data:
                if dat[0] == l:
                    nctr_by_l[l] += dat[1]
        return ','.join([f'{nctr}{lnames[l]}' for l,nctr in enumerate(nctr_by_l) if nctr > 0])

    @property
    def structure(self):
        ''' Return the contracted basis structure.

            Return:
                structure (str):
                    Contracted structure grouped by angular momentum.
        '''
        return self.structure_contracted

    @property
    def contraction_summary(self):
        ''' Return a summary of primitive-to-contracted dimensions.

            Return:
                summary (str):
                    Summary in the form "(primitive) -> [contracted]".
        '''
        return '(%s) -> [%s]' % (self.structure_primitive, self.structure_contracted)

    @property
    def angular_momenta(self):
        ''' Return the angular momentum of each channel.

            Return:
                angular_momenta (list of int):
                    Angular momenta in channel order.
        '''
        return [c.l for c in self.channels]

    @property
    def pyscf_basis(self):
        ''' Return the contracted basis in PySCF format.

            Return:
                basis (list):
                    Basis data in PySCF format.
        '''
        return self.get_pyscf_basis()

    def channel_nprim(self, channel_idx):
        ''' Return the number of primitives in one channel.

            Args:
                channel_idx (int):
                    Channel index.

            Return:
                nprim (int):
                    Number of primitives in the selected channel.
        '''
        self._check_channel_idx(channel_idx)
        return self.channels[channel_idx].nprim

    def channel_nctr(self, channel_idx):
        ''' Return the number of contractions in one channel.

            Args:
                channel_idx (int):
                    Channel index.

            Return:
                nctr (int):
                    Number of contractions in the selected channel.
        '''
        self._check_channel_idx(channel_idx)
        return self.channels[channel_idx].nctr

    def channel_l(self, channel_idx):
        ''' Return the angular momentum of one channel.

            Args:
                channel_idx (int):
                    Channel index.

            Return:
                l (int):
                    Angular momentum of the selected channel.
        '''
        self._check_channel_idx(channel_idx)
        return self.channels[channel_idx].l

    def _check_channel_idx(self, channel_idx):
        ''' Check that a channel index is in range. '''
        if channel_idx < 0 or channel_idx >= self.nchannel:
            raise IndexError('channel_idx out of range (0 ≤ channel_idx ≤ %d)'%(self.nchannel-1))

    def copy(self):
        ''' Return an independent copy of the contracted basis.

            Return:
                basis (ContractedBasis):
                    Copy containing independent contracted channels.
        '''
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
    ''' A contracted basis channel with one angular momentum.

        Args:
            l (int):
                Angular momentum.
            exponents (array_like):
                Primitive exponents.
            coefficients (array_like):
                Contraction coefficients with shape `(nprim,)` or `(nprim, nctr)`.
                A one-dimensional array defines one contraction.
    '''

    def __init__(self, l, exponents, coefficients):
        self.l = l
        self._set_data(exponents, coefficients)

    def _set_data(self, exponents, coefficients):
        ''' Validate, sort, and store exponents and coefficients.

            Args:
                exponents (array_like):
                    Primitive exponents.
                coefficients (array_like):
                    Contraction coefficients with shape `(nprim,)` or `(nprim, nctr)`.
        '''
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
        ''' Update exponents and coefficients in place.

            Args:
                exponents (array_like):
                    Primitive exponents.
                coefficients (array_like):
                    Contraction coefficients with shape `(nprim,)` or `(nprim, nctr)`.

            Return:
                self (ContractedChannel):
                    Modified contracted channel.
        '''
        self._set_data(exponents, coefficients)
        return self

    def with_data(self, exponents, coefficients):
        ''' Return a new channel with replaced contraction data.

            Args:
                exponents (array_like):
                    Primitive exponents.
                coefficients (array_like):
                    Contraction coefficients with shape `(nprim,)` or `(nprim, nctr)`.

            Return:
                channel (ContractedChannel):
                    New contracted channel containing the supplied data.
        '''
        return self.__class__(self.l, exponents, coefficients)

    @property
    def ecs(self):
        ''' Return exponents and coefficients as one array.

            Return:
                ecs (ndarray):
                    Array with exponents in the first column and coefficients in the
                    remaining columns.
        '''
        return np.column_stack((self._exponents, self._coefficients))

    @ecs.setter
    def ecs(self, value):
        ''' Set exponents and coefficients from one array.

            Args:
                value (array_like):
                    Array with exponents in the first column and coefficients in the
                    remaining columns.
        '''
        ecs = np.asarray(value, dtype=float)
        if ecs.ndim == 1: ecs = ecs.reshape(1,-1)
        elif ecs.ndim != 2:
            raise ValueError('ecs must be a one- or two-dimensional array.')
        if ecs.shape[1] < 2:
            raise ValueError('ecs must contain exponents and at least one coefficient.')
        self._set_data(ecs[:,0], ecs[:,1:])

    @property
    def nprim(self):
        ''' Return the number of primitives.

            Return:
                nprim (int):
                    Number of primitive exponents.
        '''
        return self.ecs.shape[0]

    @property
    def nctr(self):
        ''' Return the number of contractions.

            Return:
                nctr (int):
                    Number of coefficient columns.
        '''
        return self.ecs.shape[1]-1

    @property
    def nbas(self):
        ''' Return the number of radial contracted functions.

            Return:
                nbas (int):
                    Number of contractions.
        '''
        return self.nctr

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
    def exponents(self):
        ''' Return primitive exponents in ascending order.

            Return:
                exponents (ndarray):
                    Copy of the primitive exponents.
        '''
        return self._exponents.copy()

    @exponents.setter
    def exponents(self, value):
        ''' Set primitive exponents while preserving coefficients.

            Args:
                value (array_like):
                    Primitive exponents. Their length must equal `nprim`.
        '''
        self._set_data(value, self._coefficients)

    @property
    def coefficients(self):
        ''' Return coefficients ordered by ascending exponent.

            Return:
                coefficients (ndarray):
                    Copy of the contraction coefficients.
        '''
        return self._coefficients.copy()

    @coefficients.setter
    def coefficients(self, value):
        ''' Set contraction coefficients while preserving exponents.

            Args:
                value (array_like):
                    Contraction coefficients with shape `(nprim,)` or `(nprim, nctr)`.
        '''
        self._set_data(self._exponents, value)

    @property
    def structure_primitive(self):
        ''' Return the primitive structure of the channel.

            Return:
                structure (str):
                    Primitive count followed by the angular-momentum label.
        '''
        return f'{self.nprim}{lnames[self.l]}'

    @property
    def structure_contracted(self):
        ''' Return the contracted structure of the channel.

            Return:
                structure (str):
                    Contraction count followed by the angular-momentum label.
        '''
        return f'{self.nctr}{lnames[self.l]}'

    @property
    def structure(self):
        ''' Return the contracted channel structure.

            Return:
                structure (str):
                    Contracted structure of the channel.
        '''
        return self.structure_contracted

    @property
    def contraction_summary(self):
        ''' Return a summary of primitive-to-contracted dimensions.

            Return:
                summary (str):
                    Summary in the form "(primitive) -> [contracted]".
        '''
        return '(%s) -> [%s]' % (self.structure_primitive, self.structure_contracted)

    @property
    def pyscf_basis(self):
        ''' Return the contracted channel in PySCF format.

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
                    Basis data in PySCF format. An empty list is returned if no
                    exponents remain after filtering.
        '''
        basis_data = filter_basis_data_by_range([[self.l, self.ecs]], emin, emax)
        if len(basis_data) == 0:
            return []

        if sort: basis_data = sort_basis_data_by_exponents(basis_data)
        l, ecs = basis_data[0]
        return [[l] + [(x[0], *x[1:]) for x in ecs]]

    def copy(self):
        ''' Return an independent copy of the contracted channel.

            Return:
                channel (ContractedChannel):
                    Copy of the contracted channel.
        '''
        return self.__class__(self.l, self.exponents, self.coefficients)

    get_basis_str_nwchem = Channel.get_basis_str_nwchem
    dump_basis_nwchem = Channel.dump_basis_nwchem
    dump_basis = Channel.dump_basis


def parse_pyscf_basis(basis):
    ''' Convert a PySCF-format basis to internal basis data.

        Args:
            basis (list):
                Basis data in PySCF format.

        Return:
            basis_data (list of tuple):
                `(l, ecs)` pairs with copied numeric arrays.
    '''
    basis_data = []
    for b in basis:
        l = b[0]
        ecs = np.asarray(b[1:]).copy()
        if ecs.ndim == 1: ecs = ecs.reshape(1,-1)
        basis_data.append( (l, ecs) )
    return basis_data


def filter_basis_data_by_l(basis_data, keep_l=None):
    ''' Filter basis data by angular momentum.

        Args:
            basis_data (list of tuple):
                `(l, ecs)` basis-data pairs.
            keep_l (int or list of int):
                Angular momenta to keep. Default is None, which returns all data.

        Return:
            basis_data (list of tuple):
                Basis data containing the requested angular momenta.
    '''
    if keep_l is None:
        return basis_data

    keep_l = lib.to_int_list(keep_l)
    basis_data = [b for b in basis_data if b[0] in keep_l]
    return basis_data


def filter_basis_data_by_range(basis_data, emin=None, emax=None):
    ''' Filter basis data by exponent range.

        Args:
            basis_data (list of tuple):
                `(l, ecs)` basis-data pairs.
            emin/emax (float):
                Exponents outside [emin, emax] are discarded. Default is None,
                which does not impose the corresponding bound.

        Return:
            basis_data (list of tuple):
                Filtered basis data with empty channels removed.
    '''
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
    ''' Sort primitives within each basis-data channel by exponent.

        Args:
            basis_data (list of tuple):
                `(l, ecs)` basis-data pairs.
            descending (bool):
                Whether to sort in descending order. Default is True.

        Return:
            basis_data (list of tuple):
                Basis data with independently sorted and copied arrays.
    '''
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

    cgto = ContractedBasis.init_from_pyscf_basis(basis, atm=atm)
    cgto.dump_basis()
    for i in range(cgto.nchannel):
        print(cgto.channel_l(i), cgto.channel_nprim(i), cgto.channel_nctr(i))
    print(cgto.structure_primitive)
    print(cgto.structure_contracted)
    print(cgto.contraction_summary)
