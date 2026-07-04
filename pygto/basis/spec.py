import sys
import numpy as np

from pygto.basis.channel import ETB, Full



class BasisSpec:
    ''' BasisSpec is a collection of channels equipped with methods that operate on these channels. Each channel has a definite angular momentum (`channel.l`) and the basis parameters for a subset of primitives of that angular momentum. Note that multiple channels can have the same angular momentum, and therefore:

            # of channels  ≥  # of angular momentum channels

        This can be useful, e.g., separating the occupied 2p orbitals from the unoccupied 3p orbitals in Na and Mg, which in turn allow different optimization schemes to be applied to the corresponding parameters.
    '''

    def __init__(self, channels):

        # attribute
        self.channels = channels

    @classmethod
    def init_from_pyscf_basis(cls, basis, channel_type, repeat_thr=1.01, keep_l=None):
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
            angular_momenta = [l for l in angular_momenta if l in keep_l]

        channels = []
        for l in angular_momenta:
            c = Channel.init_from_pyscf_basis(l, basis, repeat_thr)
            if c.nparam > 0:
                channels.append( c )

        return cls(channels)

    def __repr__(self):
        return f'BasisSpec({self.structure})'

    # parameters
    @property
    def nparam(self):
        ''' Return total number of parameters to be optimized
        '''
        return sum([c.nparam for c in self.channels])

    @property
    def param_loc(self):
        return np.cumsum([0] + [c.nparam for c in self.channels]).astype(int)

    @property
    def parameters(self):
        if not self.channels:
            return np.asarray([], dtype=float)
        return np.hstack([c.parameters for c in self.channels])

    @parameters.setter
    def parameters(self, value):
        ''' Update parameters in place
        '''
        value = np.asarray(value, dtype=float)
        if value.size != self.nparam:
            raise ValueError(
                'Expected %d parameters, got %d' % (self.nparam, value.size)
            )

        loc = self.param_loc
        for i,(i0,i1) in enumerate(zip(loc[:-1], loc[1:])):
            self.channels[i].parameters = value[i0:i1]

    @property
    def convergence_parameters(self):
        ''' Parameters for the optimizer to calculate ∆x to check convergence,
            chosen to be log(exponents)
        '''
        return np.hstack([c.convergence_parameters for c in self.channels])

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

    @property
    def angular_momenta(self):
        ''' Return the angular momentum of each channel
        '''
        return [c.l for c in self.channels]

    @property
    def pyscf_basis(self):
        ''' Return basis in PySCF format
        '''
        basis = []
        for c in self.channels:
            basis += c.pyscf_basis
        return basis

    # method
    def copy(self):
        return self.__class__([c.copy() for c in self.channels])

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

        return self.__class__(channels)

    def merge(self, other):
        ''' Merge current BasisSpec with one or more other BasisSpec's.
        '''
        if not isinstance(other, self.__class__):
            raise TypeError('Can only merge another %s' % (self.__class__.__name__))

        overlap = set(self.angular_momenta) & set(other.angular_momenta)
        if overlap:
            raise ValueError(f'Duplicate angular momentum channels: {sorted(overlap)}')

        return self.__class__([*self.channels, *other.channels])

    def replace_channel(self, channel, channel_idx):
        if channel_idx < 0 or channel_idx >= self.nchannel:
            raise IndexError('channel_idx out of range (0 ≤ channel_idx ≤ %d)'%(self.nchannel-1))

        channels = [c.copy() for c in self.channels]
        channels[channel_idx] = channel
        return self.__class__(channels)

    def add_one_exponent_candidates(self, channel_idx, upscale=1.2, downscale=0.8, emin=0.01):
        ''' Return several new BasisSpec's with one exponent added to in a given chnanel
        '''
        if channel_idx < 0 or channel_idx >= self.nchannel:
            raise IndexError('channel_idx out of range (0 ≤ channel_idx ≤ %d)'%(self.nchannel-1))

        return [
            self.replace_channel(channel, channel_idx)
            for channel in
            self.channels[channel_idx].add_one_exponent_candidates(upscale, downscale, emin)
        ]

    def remove_one_exponent_candidates(self, channel_idx, upscale=1.2, downscale=0.8):
        ''' Return several new BasisSpec's with one exponent removed from a given chnanel
        '''
        if channel_idx < 0 or channel_idx >= self.nchannel:
            raise IndexError('channel_idx out of range (0 ≤ channel_idx ≤ %d)'%(self.nchannel-1))

        return [
            self.replace_channel(channel, channel_idx)
            for channel in
            self.channels[channel_idx].remove_one_exponent_candidates(upscale, downscale)
        ]

    def exponents_by_l(self, l):
        return self.channels[l].exponents.copy()

    def get_basis_str_nwchem(self, atm=None, header=True):
        ''' Return NWChem format basis string
        '''
        s = []
        if header:
            struct = self.structure.replace(' ', ',')
            s.append( f'#BASIS SET: ({struct}) -> [{struct}]' )

        for c in self.channels:
            s.append( c.get_basis_str(atm=atm, header=False) )

        return '\n'.join(s)

    def dump_basis_nwchem(self, des=None, atm=None, header=True):
        ''' Print NWChem format basis string to given destination
        '''
        basis_str = self.get_basis_str_nwchem(atm=atm, header=header)
        if des is None: des = sys.stdout
        des.write(basis_str + '\n')

    get_basis_str = get_basis_str_nwchem
    dump_basis = dump_basis_nwchem


if __name__ == '__main__':
    from pyscf import gto

    atm = 'C'
    mol = gto.M(atom=atm, basis='cc-pvtz', spin=None)
    basis = mol._basis[atm]

    spec = BasisSpec.init_from_pyscf_basis(basis, 'full')

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
