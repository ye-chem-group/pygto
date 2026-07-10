import sys
import numpy as np

from contextlib import contextmanager

from pygto.basis.channel import ETB, Full
from pygto.lib import StreamObject, to_int_list



class BasisSpec(StreamObject):
    ''' BasisSpec is a collection of channels equipped with methods that operate on these channels. Each channel has a definite angular momentum (`channel.l`) and the basis parameters for a subset of primitives of that angular momentum. Note that multiple channels can have the same angular momentum, and therefore:

            # of channels  ≥  # of angular momentum channels

        This can be useful, e.g., separating the occupied 2p orbitals from the unoccupied 3p orbitals in Na and Mg, which in turn allow different optimization schemes to be applied to the corresponding parameters.
    '''

    _keys = {
        'atm', 'active_channel'
    }

    def __init__(self, channels):

        # attribute from initialization
        self.channels = channels

        # attribute with default
        self.atm = None
        self._active_channel = None

    @classmethod
    def init_from_pyscf_basis(cls, basis, channel_type='full', repeat_thr=1.01, keep_l=None,
                              emin=None, emax=None, atm=None):
        ''' Init from PySCF basis.

            Note:
                - The input basis will be fully decontracted.
                - Exponents of same angular momentum will be merged into a single channel.
                - Repeated exponents (defined as ratio < `repeat_thr`) will be removed.
        '''
        try:
            Channel = {
                'etb': ETB,
                'full': Full
            }[channel_type.lower()]
        except:
            raise ValueError('Channel type must be "etb" or "full" (case insensitive).')

        angular_momenta = sorted(list(set([int(b[0]) for b in basis])))
        if keep_l is not None:
            keep_l = to_int_list(keep_l)
            angular_momenta = [l for l in angular_momenta if l in keep_l]

        channels = []
        for l in angular_momenta:
            c = Channel.init_from_pyscf_basis(l, basis, repeat_thr, emin, emax)
            if c.nparam > 0:
                channels.append( c )

        return cls(channels).set(atm=atm)

    def convert_to(self, channel_type):
        ''' Convert a copy of current BasisSpec into specified channel type(s).
        '''
        new = self.copy()
        new.convert_to_(channel_type)
        return new

    def convert_to_(self, channel_type):
        ''' Convert current BasisSpec in place into specified channel type(s).
        '''
        if isinstance(channel_type, str):
            channel_type = [channel_type] * self.nchannel
        elif isinstance(channel_type, (list,tuple)):
            if len(channel_type) != self.nchannel:
                raise ValueError('Length of `channel_type` does not match `nchannel` (%d != %d)'
                                 % (len(channel_type), self.nchannel))
            if not all([isinstance(c, str) for c in channel_type]):
                raise TypeError('All elements in `channel_type` must be str.')
        else:
            raise TypeError('`channel_type` must be str of list/tuple of str.')

        self.channels = [
            c.convert_to(ct)
            for ct, c in zip(channel_type, self.channels)
        ]

        return self

    def __repr__(self):
        return f'BasisSpec({self.structure})'

    # parameters
    @property
    def nparam(self):
        ''' Return total number of parameters to be optimized
        '''
        mask = self.get_active_mask()
        return sum([c.nparam for c,m in zip(self.channels,mask) if m])

    @property
    def param_loc(self):
        mask = self.get_active_mask()
        return np.cumsum([0] + [c.nparam for c,m in zip(self.channels,mask) if m]).astype(int)

    @property
    def parameters(self):
        if not self.channels:
            return np.asarray([], dtype=float)
        mask = self.get_active_mask()
        return np.hstack([c.parameters for c,m in zip(self.channels,mask) if m])

    @parameters.setter
    def parameters(self, value):
        ''' Update parameters in place
        '''
        value = np.asarray(value, dtype=float)
        if value.size != self.nparam:
            raise ValueError(
                'Expected %d parameters, got %d' % (self.nparam, value.size)
            )

        active_channel_idx = np.where(self.get_active_mask())[0]
        loc = self.param_loc
        for i,(i0,i1) in enumerate(zip(loc[:-1], loc[1:])):
            idx = active_channel_idx[i]
            self.channels[idx].parameters = value[i0:i1]

    @property
    def convergence_parameters(self):
        ''' Parameters for the optimizer to calculate ∆x to check convergence,
            chosen to be log(exponents)
        '''
        mask = self.get_active_mask()
        return np.hstack([c.convergence_parameters for c,m in zip(self.channels,mask) if m])

    def parameter_jacobian(self):
        ''' Return d(physical parameters) / d(parameters).
        '''
        raise NotImplementedError

    def with_parameters(self, value):
        ''' Return a new BasisSpec with updated parameters
        '''
        spec = self.copy()
        spec.parameters = value
        return spec

    @property
    def active_channel(self):
        return self._active_channel

    @active_channel.setter
    def active_channel(self, value):
        self.set_active_channel(value)

    @property
    def active_l(self):
        if self.active_channel is None:
            return None
        return sorted(list(set([self.channels[i].l for i in self.active_channel])))

    @active_l.setter
    def active_l(self, value):
        self.set_active_l(value)

    def set_active_channel(self, active_channel=None):
        ''' Set active channels by channel index.
            Use `None` to reset and activate all channels.
        '''
        if active_channel is None:
            self._active_channel = None  # reset and activate all channels
            return

        active_channel = to_int_list(active_channel)

        if len(active_channel) == 0:
            raise ValueError('Channel index must not be empty.')

        if not set(active_channel).issubset(set(range(self.nchannel))):
            raise ValueError('Channel index out of range.')

        self._active_channel = sorted(list(set(active_channel)))

    def set_active_l(self, active_l=None):
        ''' Set active channels by angular momentum.
            Use `None` to reset and activate all channels.
        '''
        if active_l is None:
            self._active_channel = None  # reset and activate all channels
            return

        active_l = to_int_list(active_l)

        if len(active_l) == 0:
            raise ValueError('Angular momenta must not be empty.')

        if not set(active_l).issubset(set(self.angular_momenta)):
            raise ValueError('Angular momenta out of range.')

        active_l = sorted(list(set(active_l)))

        self._active_channel = [i for i,c in enumerate(self.channels) if c.l in active_l]

    @contextmanager
    def temporary_active_channel(self, active_channel=None):
        old_active_channel = (
            None if self.active_channel is None else self.active_channel.copy()
        )
        try:
            self.set_active_channel(active_channel)
            yield self
        finally:
            self.set_active_channel(old_active_channel)

    @contextmanager
    def temporary_active_l(self, active_l=None):
        old_active_channel = (
            None if self.active_channel is None else self.active_channel.copy()
        )
        try:
            self.set_active_l(active_l)
            yield self
        finally:
            self.set_active_channel(old_active_channel)

    def get_active_mask(self, active_channel=None):
        ''' Return a boolean array specifying which channels are active.
        '''
        if active_channel is None:
            active_channel = self._active_channel
        else:
            active_channel = to_int_list(active_channel)

        if active_channel is None:
            mask = np.ones(self.nchannel, dtype=bool)
        else:
            mask = np.zeros(self.nchannel, dtype=bool)
            mask[active_channel] = True

        return mask

    # property
    @property
    def structure(self):
        ''' Return basis structureure (e.g., "[4s,3p,2d,1f]")
        '''
        return ' '.join([c.structure for c in self.channels])

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

    def _check_channel_idx(self, channel_idx):
        if channel_idx < 0 or channel_idx >= self.nchannel:
            raise IndexError('channel_idx out of range (0 ≤ channel_idx ≤ %d)'%(self.nchannel-1))

    @property
    def angular_momenta(self):
        ''' Return the angular momentum of each channel
        '''
        return [c.l for c in self.channels]

    @property
    def pyscf_basis(self):
        ''' Return basis in PySCF format
        '''
        return self.get_pyscf_basis()

    # method
    def copy(self):
        new = self.__class__([c.copy() for c in self.channels])
        for k in self._keys:
            setattr(new, k, getattr(self, k))
        return new

    def merge_angular_momentum(self):
        ''' Merge channels of same angular momentum.

            Note: A RuntimeError will be thrown if the channels to be merged are of different type.
        '''
        LMAX = 10
        channels_pool = [None] * LMAX
        for c in self.channels:
            l = c.l
            if channels_pool[l] is None:
                channels_pool[l] = []
            channels_pool[l] += [c]

        channels = []
        for l in range(LMAX):
            if channels_pool[l] is None: continue

            c = channels_pool[l][0].copy()
            if len(channels_pool[l]) == 1:
                channels.append( c )
            else:
                channels.append( c.merge(channels_pool[l][1:]) )

        new = self.copy()
        new.channels = channels
        return new

    def merge(self, other):
        ''' Merge current BasisSpec with one or more other BasisSpec's.
        '''
        if not isinstance(other, self.__class__):
            raise TypeError('Can only merge another %s' % (self.__class__.__name__))

        if self.atm != other.atm:
            raise ValueError('Cannot merge two specs with different `atm`.')

        overlap = set(self.angular_momenta) & set(other.angular_momenta)
        if overlap:
            raise ValueError(f'Duplicate angular momentum channels: {sorted(overlap)}')

        new = self.copy()
        new.channels = [c.copy() for c in [*self.channels, *other.channels]]
        return new

    def replace_channel(self, channel, channel_idx):
        ''' Replace a specific channel by channel index
        '''
        self._check_channel_idx(channel_idx)

        new = self.copy()
        new.channels[channel_idx] = channel
        return new

    def add_one_exponent_candidates(self, channel_idx, upscale=1.2, downscale=0.8, emin=0.01):
        ''' Return several new BasisSpec's with one exponent added to in a given chnanel
        '''
        self._check_channel_idx(channel_idx)

        return [
            self.replace_channel(channel, channel_idx)
            for channel in
            self.channels[channel_idx].add_one_exponent_candidates(upscale, downscale, emin)
        ]

    def remove_one_exponent_candidates(self, channel_idx, upscale=1.2, downscale=0.8):
        ''' Return several new BasisSpec's with one exponent removed from a given chnanel
        '''
        self._check_channel_idx(channel_idx)

        return [
            self.replace_channel(channel, channel_idx)
            for channel in
            self.channels[channel_idx].remove_one_exponent_candidates(upscale, downscale)
        ]

    def get_ratio_penalty(self, ratio_min=None, strength=None):
        ''' Penalty on two exponents being too close within a channel.
        '''
        return sum([c.get_ratio_penalty(ratio_min, strength) for c in self.channels])

    def exponents_by_l(self, l):
        return self.channels[l].exponents.copy()

    def remove_one_exponent_candidates_rigid(self, channel_idx, emin=None, emax=None):
        ''' Return a list of new BasisSpec's with one exponent removed from given channel.

            Use emin and emax for select the exponent window from which you want the
            exponents to be removed. Default is None, which means no window is applied.
        '''
        self._check_channel_idx(channel_idx)

        return [
            self.replace_channel(channel, channel_idx)
            for channel in
            self.channels[channel_idx].remove_one_exponent_candidates_rigid(emin, emax)
        ]

    def filter_channel_by_index_(self, channel_idx, exponent_idx):
        ''' Filter a channel in place by exponent index
        '''
        self._check_channel_idx(channel_idx)
        self.channels[channel_idx].filter_by_index_(exponent_idx)

    def filter_channel_by_index(self, channel_idx, exponent_idx):
        ''' Return a new BasisSpec where a channel is filtered by exponent index
        '''
        new = self.copy()
        new.filter_channel_by_index_(channel_idx, exponent_idx)
        return new

    def filter_channel_by_exponent_range_(self, channel_idx, emin=None, emax=None):
        ''' Filter a channel in place by [emin, emax]
        '''
        self._check_channel_idx(channel_idx)
        self.channels[channel_idx].filter_by_exponent_range_(emin, emax)

    def filter_channel_by_exponent_range(self, channel_idx, emin=None, emax=None):
        ''' Return a new BasisSpec where a channel is filtered by [emin, emax]
        '''
        new = self.copy()
        new.filter_channel_by_exponent_range_(channel_idx, emin, emax)
        return new

    def get_pyscf_basis(self, keep_l=None, emin=None, emax=None):
        ''' Return basis set in PySCF format

            Args:
                keep_l (int or list of int):
                    Specifying which angular momentum channel to keep.
                    Default is None, which keeps all channels.

                emin/emax (float):
                    Only exponents within [emin, emax] will be kept.
                    Default is None, which does not filter.

            Note:
                keep_l is independent of active_l, i.e.,
                ```
                    with spec.temporary_active_l([0,1]):
                        basis = spec.get_pyscf_basis()
                ```
                still gives the full basis set with all channels.
        '''
        if keep_l is None:
            keep_l = self.angular_momenta

        keep_l = to_int_list(keep_l)

        basis = []
        for c in self.channels:
            if c.l in keep_l:
                basis += c.get_pyscf_basis(emin, emax)

        return basis

    def get_basis_str_nwchem(self, atm=None, header=True):
        ''' Return NWChem format basis string
        '''
        if atm is None: atm = self.atm

        s = []
        if header:
            struct = self.structure.replace(' ', ',')
            s.append( f'#BASIS SET: ({struct}) -> [{struct}]' )

        for c in self.channels:
            s.append( c.get_basis_str(atm=atm, header=False) )

        return '\n'.join(s)

    def dump_basis_nwchem(self, stdout=None, atm=None, header=True):
        ''' Print NWChem format basis string to given destination
        '''
        basis_str = self.get_basis_str_nwchem(atm=atm, header=header)
        if stdout is None: stdout = self.stdout
        stdout.write(basis_str + '\n')

    get_basis_str = get_basis_str_nwchem
    dump_basis = dump_basis_nwchem


if __name__ == '__main__':
    from pyscf import gto

    atm = 'C'
    mol = gto.M(atom=atm, basis='cc-pvtz', spin=None)
    basis = mol._basis[atm]

    spec = BasisSpec.init_from_pyscf_basis(basis, atm=atm)

    print(spec)
    print(spec.structure)
    print(spec.nao)
    print(spec.nbas)
    print(spec.nparam)
    print(spec.pyscf_basis)
    print(spec.exponents_by_l(0))
    print(spec.exponents_by_l(1))
    # print(spec.get_basis_str(atm=atm))
    spec.dump_basis(atm=atm)

    # from zflow.pyscf_helper import dump_basis
    # dump_basis({atm:spec.pyscf_basis})
