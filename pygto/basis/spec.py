import sys
import h5py
import numpy as np

from contextlib import contextmanager

from pygto.basis.channel import ETB, Full
from pygto.lib import StreamObject, to_int_list, chkfile_helper
from pygto.lib import load_basis_nwchem, dump_basis_nwchem, get_basis_str_nwchem, get_named_basis


class BasisSpec(StreamObject):
    ''' A collection of channels that defines an optimizable basis set.

        Each channel has a definite angular momentum (`channel.l`) and contains
        the parameters for a subset of primitives of that angular momentum.
        Multiple channels may have the same angular momentum. This can be useful,
        for example, for optimizing occupied and unoccupied shells separately.

        Args:
            channels (list of Channel):
                Basis-set channels.

        Attributes:
            atm (str):
                Atomic symbol associated with the basis. Default is None.
            active_channel (list of int):
                Indices of channels included in parameter operations. Default is None,
                which activates all channels.
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
    def init_from_named_basis(cls, name, atm, **kwargs):
        ''' Initialize a BasisSpec object from a named basis set.

            Args:
                name (str):
                    Basis-set name recognized by Basis Set Exchange.
                atm (str):
                    Atomic symbol of the basis to load.
                kwargs (dict):
                    Additional arguments passed to `init_from_pyscf_basis`.

            Return:
                spec (BasisSpec):
                    BasisSpec object initialized from the named basis.
        '''
        basis = get_named_basis(name, atm)
        return cls.init_from_pyscf_basis(basis, atm=atm, **kwargs)

    @classmethod
    def init_from_nwchem_basis(cls, basis_str_or_file, atm, **kwargs):
        ''' Initialize a BasisSpec object from an NWChem-format basis.

            Args:
                basis_str_or_file (str):
                    NWChem basis data or the path to an NWChem basis file.
                atm (str):
                    Atomic symbol of the basis to load.
                kwargs (dict):
                    Additional arguments passed to :func:`init_from_pyscf_basis`.

            Return:
                spec (BasisSpec):
                    BasisSpec object initialized from the NWChem basis.
        '''
        basis = load_basis_nwchem(basis_str_or_file, atm)
        return cls.init_from_pyscf_basis(basis, atm=atm, **kwargs)

    @classmethod
    def init_from_pyscf_basis(cls, basis, channel_type='full', repeat_thr=1.01, keep_l=None,
                              emin=None, emax=None, atm=None):
        ''' Initialize a BasisSpec object from a PySCF-format basis.

            Args:
                basis (list):
                    Basis data in PySCF format.
                channel_type (str):
                    Type of channels to construct. Accepted values are "etb" and
                    "full" (case insensitive). Default is "full".
                repeat_thr (float):
                    Exponents whose ratio is smaller than this threshold are treated
                    as repeated and thus discarded. Default is 1.01.
                keep_l (int or list of int):
                    Angular momenta to keep. Default is None, which keeps all channels.
                emin/emax (float):
                    Exponents outside [emin, emax] are discarded. Default is None,
                    which does not impose the corresponding bound.
                atm (str):
                    Atomic symbol associated with the basis. Default is None.

            Note:
                - The input basis will be fully decontracted.
                - Exponents of same angular momentum will be merged into a single channel.
                - Repeated exponents (defined as ratio < `repeat_thr`) will be removed.

            Return:
                spec (BasisSpec):
                    BasisSpec object initialized from the PySCF basis.
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
        ''' Convert a copy of the BasisSpec to specified channel types.

            Args:
                channel_type (str or list of str):
                    Channel type for all channels or one type per channel. Accepted
                    values are "etb" and "full" (case insensitive).

            Return:
                spec (BasisSpec):
                    A converted copy of the BasisSpec.
        '''
        new = self.copy()
        new.convert_to_(channel_type)
        return new

    def convert_to_(self, channel_type):
        ''' Convert channels to specified types in place.

            Args:
                channel_type (str or list of str):
                    Channel type for all channels or one type per channel. Accepted
                    values are "etb" and "full" (case insensitive).

            Return:
                self (BasisSpec):
                    The modified BasisSpec object.
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
        ''' Return a concise representation of the BasisSpec. '''
        return f'BasisSpec({self.structure})'

    # parameters
    @property
    def nparam(self):
        ''' Return the total number of parameters in active channels.

            Return:
                nparam (int):
                    Number of active parameters.
        '''
        mask = self.get_active_mask()
        return sum([c.nparam for c,m in zip(self.channels,mask) if m])

    @property
    def param_loc(self):
        ''' Return parameter offsets for active channels.

            Return:
                loc (ndarray of int):
                    Cumulative parameter offsets, including the initial zero.
        '''
        mask = self.get_active_mask()
        return np.cumsum([0] + [c.nparam for c,m in zip(self.channels,mask) if m]).astype(int)

    @property
    def parameters(self):
        ''' Return parameters from active channels as a one-dimensional array.

            Return:
                parameters (ndarray):
                    Concatenated channel parameters.
        '''
        if not self.channels:
            return np.asarray([], dtype=float)
        mask = self.get_active_mask()
        return np.hstack([c.parameters for c,m in zip(self.channels,mask) if m])

    @parameters.setter
    def parameters(self, value):
        ''' Update parameters in active channels in place.

            Args:
                value (array_like):
                    New parameters, ordered according to `param_loc`.
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
        ''' Return parameters used by optimizers to check convergence.

            The convergence parameters are chosen to be the logarithms of the
            exponents in active channels.

            Return:
                parameters (ndarray):
                    Concatenated convergence parameters.
        '''
        mask = self.get_active_mask()
        return np.hstack([c.convergence_parameters for c,m in zip(self.channels,mask) if m])

    def parameter_jacobian(self):
        ''' Return the Jacobian of physical parameters with respect to parameters.

            Return:
                jacobian (ndarray):
                    Derivatives of physical parameters with respect to optimization
                    parameters.
        '''
        raise NotImplementedError

    def with_parameters(self, value):
        ''' Return a copy with updated parameters.

            Args:
                value (array_like):
                    New parameters for active channels.

            Return:
                spec (BasisSpec):
                    A copy with the updated parameters.
        '''
        spec = self.copy()
        spec.parameters = value
        return spec

    def with_channels(self, channels):
        ''' Return a copy with replaced channels.

            Args:
                channels (list of Channel):
                    Replacement basis-set channels.

            Return:
                spec (BasisSpec):
                    A copy containing the replacement channels.
        '''
        spec = self.copy()
        spec.channels = channels
        return spec

    @property
    def active_channel(self):
        ''' Return indices of active channels.

            Return:
                active_channel (list of int or None):
                    Active channel indices. None means that all channels are active.
        '''
        return self._active_channel

    @active_channel.setter
    def active_channel(self, value):
        ''' Set active channels by channel index. '''
        self.set_active_channel(value)

    @property
    def active_l(self):
        ''' Return angular momenta represented by active channels.

            Return:
                active_l (list of int or None):
                    Active angular momenta. None means that all channels are active.
        '''
        if self.active_channel is None:
            return None
        return sorted(list(set([self.channels[i].l for i in self.active_channel])))

    @active_l.setter
    def active_l(self, value):
        ''' Set active channels by angular momentum. '''
        self.set_active_l(value)

    def set_active_channel(self, active_channel=None):
        ''' Set active channels by channel index.

            Args:
                active_channel (int or list of int):
                    Channel indices to activate. Default is None, which activates all
                    channels.
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

            Args:
                active_l (int or list of int):
                    Angular momenta to activate. Default is None, which activates all
                    channels.
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
        ''' Temporarily set active channels by channel index.

            Usage:
            ```
                with spec.temporary_active_channel(active_channel):
                    # do sth with spec
            ```

            Args:
                active_channel (int or list of int):
                    Channel indices to activate. Default is None, which activates all
                    channels within the context.

            Yields:
                self (BasisSpec):
                    The BasisSpec with the temporary active-channel selection.
        '''
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
        ''' Temporarily set active channels by angular momentum.

            Usage:
            ```
                with spec.temporary_active_l(active_l):
                    # do sth with spec
            ```

            Args:
                active_l (int or list of int):
                    Angular momenta to activate. Default is None, which activates all
                    channels within the context.

            Yields:
                self (BasisSpec):
                    The BasisSpec with the temporary angular-momentum selection.
        '''
        old_active_channel = (
            None if self.active_channel is None else self.active_channel.copy()
        )
        try:
            self.set_active_l(active_l)
            yield self
        finally:
            self.set_active_channel(old_active_channel)

    def get_active_mask(self, active_channel=None):
        ''' Return a boolean mask specifying active channels.

            Args:
                active_channel (int or list of int):
                    Channel indices used to construct the mask. Default is None, which
                    uses the current active-channel selection.

            Return:
                mask (ndarray of bool):
                    Boolean mask with one entry per channel.
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
        ''' Return the basis structure, for example, "4s,3p,2d,1f".

            Return:
                structure (str):
                    Space-separated channel structures.
        '''
        return ','.join([c.structure for c in self.channels if c.nbas > 0])

    @property
    def nao(self):
        ''' Return the total number of basis functions including m components.

            Return:
                nao (int):
                    Number of spherical atomic orbitals.
        '''
        return sum([c.nao for c in self.channels])

    @property
    def nbas(self):
        ''' Return the total number of basis functions excluding m components.

            Return:
                nbas (int):
                    Number of radial basis functions.
        '''
        return sum([c.nbas for c in self.channels])

    @property
    def nchannel(self):
        ''' Return the number of channels.

            Return:
                nchannel (int):
                    Number of basis-set channels.
        '''
        return len(self.channels)

    def _check_channel_idx(self, channel_idx):
        ''' Check that a channel index is in range. '''
        if channel_idx < 0 or channel_idx >= self.nchannel:
            raise IndexError('channel_idx out of range (0 ≤ channel_idx ≤ %d)'%(self.nchannel-1))

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
        ''' Return the basis in PySCF format.

            Return:
                basis (list):
                    Basis data in PySCF format.
        '''
        return self.get_pyscf_basis()

    # method
    def copy(self):
        ''' Return an independent copy of the BasisSpec.

            Return:
                spec (BasisSpec):
                    A copy containing copies of all channels.
        '''
        new = self.__class__([c.copy() for c in self.channels])
        for k in self._keys:
            setattr(new, k, getattr(self, k))
        return new

    def merge_angular_momentum(self):
        ''' Merge channels having the same angular momentum.

            Return:
                spec (BasisSpec):
                    A copy with same-angular-momentum channels merged.

            Note:
                Channels can only be merged when their channel types are compatible.
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
        ''' Merge with another BasisSpec containing distinct angular momenta.

            Args:
                other (BasisSpec):
                    BasisSpec to merge with the current object.

            Return:
                spec (BasisSpec):
                    A merged copy of both BasisSpec objects.
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

    def replace_channel(self, channel_idx, channel):
        ''' Return a copy with one channel replaced.

            Args:
                channel_idx (int):
                    Index of the channel to replace.
                channel (Channel):
                    Replacement channel.

            Return:
                spec (BasisSpec):
                    A copy containing the replacement channel.
        '''
        self._check_channel_idx(channel_idx)

        new = self.copy()
        new.channels[channel_idx] = channel
        return new

    def replace_channel_(self, channel_idx, channel):
        ''' Replace one channel in place.

            Args:
                channel_idx (int):
                    Index of the channel to replace.
                channel (Channel):
                    Replacement channel.
        '''
        self._check_channel_idx(channel_idx)
        self.channels[channel_idx] = channel

    def add_one_exponent_candidates(self, channel_idx, upscale=1.2, downscale=0.8, emin=0.01):
        ''' Return candidate BasisSpec objects with one exponent added.

            Args:
                channel_idx (int):
                    Index of the channel to modify.
                upscale (float):
                    Scaling factor applied to existing exponents when adding a
                    diffuse exponent. Default is 1.2.
                downscale (float):
                    Scaling factor applied to existing exponents when adding a tight
                    exponent. Default is 0.8.
                emin (float):
                    Lower bound for the most diffuse candidate. Default is 0.01.

            Return:
                specs (list of BasisSpec):
                    Candidate BasisSpec objects.
        '''
        self._check_channel_idx(channel_idx)

        return [
            self.replace_channel(channel_idx, channel)
            for channel in
            self.channels[channel_idx].add_one_exponent_candidates(upscale, downscale, emin)
        ]

    def remove_one_exponent_candidates(self, channel_idx, upscale=1.2, downscale=0.8):
        ''' Return candidate BasisSpec objects with one exponent removed.

            Args:
                channel_idx (int):
                    Index of the channel to modify.
                upscale (float):
                    Scaling factor applied after removing the tightest exponent.
                    Default is 1.2.
                downscale (float):
                    Scaling factor applied after removing the most diffuse exponent.
                    Default is 0.8.

            Return:
                specs (list of BasisSpec):
                    Candidate BasisSpec objects.
        '''
        self._check_channel_idx(channel_idx)

        return [
            self.replace_channel(channel_idx, channel)
            for channel in
            self.channels[channel_idx].remove_one_exponent_candidates(upscale, downscale)
        ]

    def get_ratio_penalty(self, ratio_min=None, strength=None):
        ''' Return the penalty for exponents that are too close within channels.

            Args:
                ratio_min (float):
                    Minimum desired ratio between adjacent exponents. Default is None.
                strength (float):
                    Penalty strength. Default is None.

            Return:
                penalty (float):
                    Sum of ratio penalties over all channels.
        '''
        return sum([c.get_ratio_penalty(ratio_min, strength) for c in self.channels])

    def exponents_by_l(self, l):
        ''' Return exponents for an angular-momentum channel.

            Args:
                l (int):
                    Angular momentum and corresponding channel index.

            Return:
                exponents (ndarray):
                    A copy of the channel exponents.

            Note:
                This method assumes that channel index and angular momentum coincide.
        '''
        return self.channels[l].exponents.copy()

    def remove_one_exponent_candidates_rigid(self, channel_idx, emin=None, emax=None):
        ''' Return candidates with one exponent rigidly removed from a channel.

            Args:
                channel_idx (int):
                    Index of the channel to modify.
                emin/emax (float):
                    The channel is first restricted to exponents within [emin, emax].
                    Default is None, which does not impose the corresponding bound.

            Return:
                specs (list of BasisSpec):
                    Candidate BasisSpec objects.
        '''
        self._check_channel_idx(channel_idx)

        return [
            self.replace_channel(channel_idx, channel)
            for channel in
            self.channels[channel_idx].remove_one_exponent_candidates_rigid(emin, emax)
        ]

    def filter_channel_by_index_(self, channel_idx, exponent_idx):
        ''' Filter one channel by exponent index in place.

            Args:
                channel_idx (int):
                    Index of the channel to filter.
                exponent_idx (int or list of int):
                    Exponent indices to retain.
        '''
        self._check_channel_idx(channel_idx)
        self.channels[channel_idx].filter_by_index_(exponent_idx)

    def filter_channel_by_index(self, channel_idx, exponent_idx):
        ''' Return a copy with one channel filtered by exponent index.

            Args:
                channel_idx (int):
                    Index of the channel to filter.
                exponent_idx (int or list of int):
                    Exponent indices to retain.

            Return:
                spec (BasisSpec):
                    A filtered copy of the BasisSpec.
        '''
        new = self.copy()
        new.filter_channel_by_index_(channel_idx, exponent_idx)
        return new

    def filter_channel_by_exponent_range_(self, channel_idx, emin=None, emax=None):
        ''' Filter one channel by exponent range in place.

            Args:
                channel_idx (int):
                    Index of the channel to filter.
                emin/emax (float):
                    Exponents outside [emin, emax] are discarded. Default is None,
                    which does not impose the corresponding bound.
        '''
        self._check_channel_idx(channel_idx)
        self.channels[channel_idx].filter_by_exponent_range_(emin, emax)

    def filter_channel_by_exponent_range(self, channel_idx, emin=None, emax=None):
        ''' Return a copy with one channel filtered by exponent range.

            Args:
                channel_idx (int):
                    Index of the channel to filter.
                emin/emax (float):
                    Exponents outside [emin, emax] are discarded. Default is None,
                    which does not impose the corresponding bound.

            Return:
                spec (BasisSpec):
                    A filtered copy of the BasisSpec.
        '''
        new = self.copy()
        new.filter_channel_by_exponent_range_(channel_idx, emin, emax)
        return new

    def get_pyscf_basis(self, keep_l=None, emin=None, emax=None, sort=True):
        ''' Return the basis set in PySCF format.

            Args:
                keep_l (int or list of int):
                    Specifying which angular momentum channel to keep.
                    Default is None, which keeps all channels.

                emin/emax (float):
                    Only exponents within [emin, emax] will be kept.
                    Default is None, which does not filter.

                sort (bool):
                    Whether to sort the exponents in descending order. Default is True.

            Note:
                keep_l is independent of active_l, i.e.,
                ```
                    with spec.temporary_active_l([0,1]):
                        basis = spec.get_pyscf_basis()
                ```
                still gives the full basis set with all channels.

            Return:
                Basis data in PySCF format.
        '''
        if keep_l is None:
            keep_l = self.angular_momenta

        keep_l = to_int_list(keep_l)

        basis = []
        for c in self.channels:
            if c.l in keep_l:
                basis += c.get_pyscf_basis(emin, emax, sort=sort)

        return basis

    def get_basis_str_nwchem(self, atm=None, header=True, sort=True):
        ''' Return the basis as an NWChem-format string.

            Args:
                atm (str):
                    Atomic symbol printed in the basis. Default is None, which uses "X".
                header (bool):
                    Whether to include the basis header. Default is True.
                sort (bool):
                    Whether to sort angular momenta and exponents in canonical order.
                    Default is True.

            Return:
                basis_str (str):
                    Basis data in NWChem format.
        '''
        return get_basis_str_nwchem(
            self.get_pyscf_basis(sort=sort), atm, header, sort
        )

    def dump_basis_nwchem(self, stdout=None, atm=None, header=True, sort=True):
        ''' Write the basis in NWChem format.

            Args:
                stdout (file-like object):
                    Destination for the basis data. Default is None, which uses
                    `sys.stdout`.
                atm (str):
                    Atomic symbol printed in the basis. Default is None, which uses
                    `self.atm`.
                header (bool):
                    Whether to include the basis header. Default is True.
                sort (bool):
                    Whether to sort angular momenta and exponents in canonical order.
                    Default is True.
        '''
        if atm is None: atm = self.atm
        dump_basis_nwchem(
            self.get_pyscf_basis(sort=sort), stdout, atm, header, sort
        )

    def dump_channel_basis(self, channel_idx, stdout=None, atm=None, header=True, sort=True):
        ''' Write one channel in NWChem format.

            Args:
                channel_idx (int):
                    Index of the channel to write.
                stdout (file-like object):
                    Destination for the basis data. Default is None, which uses
                    `self.stdout`.
                atm (str):
                    Atomic symbol printed in the basis. Default is None, which uses
                    `self.atm`.
                header (bool):
                    Whether to include the basis header. Default is True.
                sort (bool):
                    Whether to sort exponents in descending order. Default is True.
        '''
        self._check_channel_idx(channel_idx)
        if stdout is None: stdout = self.stdout
        if atm is None: atm = self.atm
        return self.channels[channel_idx].dump_basis(
            stdout=stdout, atm=atm, header=header, sort=sort,
        )

    get_basis_str = get_basis_str_nwchem
    dump_basis = dump_basis_nwchem

    def dump_chkfile(self, chkfile, prefix=None):
        ''' Save BasisSpec data to chkfile.

            Args:
                chkfile (str):
                    Path to chkfile.
                prefix (str):
                    Key in h5py file. Default is None, which uses "spec".
                    - `spec.atm` is stored as "[prefix]/atm".
                    - `spec.channels` is stored as "[prefix]/channel_i" for
                      i = 0, 1, ...
        '''
        if prefix is None: prefix = 'spec'

        # Remove old channels when overwriting a spec with fewer channels.
        with h5py.File(chkfile, 'a') as f:
            if prefix in f:
                del f[prefix]
            f.require_group(prefix)

        if self.atm is not None:
            chkfile_helper.dump(chkfile, f'{prefix}/atm', self.atm)
        for i,c in enumerate(self.channels):
            c.dump_chkfile(chkfile, f'{prefix}/channel_{i}')

    @classmethod
    def init_from_chkfile(cls, chkfile, prefix=None, channel_type=None):
        ''' Initialize a BasisSpec object from chkfile.

            Args:
                chkfile (str):
                    Path to chkfile.
                prefix (str):
                    Key in h5py file. Default is None, which uses "spec".
                    - `spec.atm` is loaded from "[prefix]/atm". If it does not exist,
                      `spec.atm` is set to None.
                    - `spec.channels` is loaded from "[prefix]/channel_i" for i = 0, 1, ...
                channel_type (str):
                    Channel type to which loaded channels are converted. Accepted
                    values are "etb" and "full" (case insensitive). Default is None,
                    which uses the saved channel type without conversion.

            Return:
                spec (BasisSpec):
                    BasisSpec object initialized using data saved in `chkfile`.
        '''
        channel_types = {
            'etb': ETB,
            'full': Full
        }
        if channel_type is not None:
            try:
                channel_type = channel_type.lower()
                channel_types[channel_type]
            except (AttributeError, KeyError):
                raise ValueError('Channel type must be "etb" or "full" (case insensitive).')

        if prefix is None: prefix = 'spec'
        try:
            atm = chkfile_helper.load(chkfile, f'{prefix}/atm')
        except KeyError:
            atm = None

        with h5py.File(chkfile, 'r') as f:
            channel_idxs = sorted([
                int(x.replace('channel_', ''))
                for x in list(f[prefix]) if x.startswith('channel_')
            ])

        channels = []
        for channel_idx in channel_idxs:
            channel_prefix = f'{prefix}/channel_{channel_idx}'
            saved_channel_type = chkfile_helper.load(chkfile, f'{channel_prefix}/type')
            try:
                Channel = channel_types[saved_channel_type.lower()]
            except (AttributeError, KeyError):
                raise ValueError('Unknown channel type in chkfile: %s' % saved_channel_type)
            channels.append(Channel.init_from_chkfile(chkfile, channel_prefix))

        spec = cls(channels).set(atm=atm)
        if channel_type is not None:
            spec.convert_to_(channel_type)
        return spec


if __name__ == '__main__':
    atm = 'C'
    named_basis = 'cc-pvtz'

    spec = BasisSpec.init_from_named_basis(
        named_basis, atm,
        keep_l=[0,1,2],         # keeping s, p, and d
        emin=0.15, emax=1500.,  # keeping only exponents in [0.15, 1500.]
    )

    print(spec)
    print('structure= %s' % (spec.structure))
    print('nao= %d' % (spec.nao))
    print('nbas= %d' % (spec.nbas))
    print('nparam= %d' % (spec.nparam))
    print('angular momenta= %s' % (str(spec.angular_momenta)))
    print('pyscf basis= %s' % (str(spec.pyscf_basis)))
    print('')
    spec.dump_basis(atm=atm)
