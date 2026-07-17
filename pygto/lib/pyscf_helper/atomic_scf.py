import numpy as np


def _prepare_atomic_config(mf, config):
    ''' Validate an atomic SCF configuration and collect angular-momentum indices.

        Args:
            mf (pyscf.scf.hf.SCF):
                PySCF atomic SCF or Kohn-Sham object.
            config (array_like):
                Electron counts `[ns, np, nd, nf]` for restricted methods or an
                `(alpha, beta)` pair of such arrays for open-shell methods.

        Return:
            method (str):
                Occupation type: "rhf", "rohf", or "uhf".
            config_alpha/config_beta (ndarray):
                Spin-resolved electron counts by angular momentum.
            ao_l (ndarray):
                Angular momentum assigned to each AO.
            ao_idx (dict):
                AO indices grouped by angular momentum.
    '''
    import numbers

    from pyscf import dft, scf

    mol = mf.mol
    if mol.natm != 1:
        raise ValueError('Only single-atom SCF object is supported.')
    if mol.cart:
        raise ValueError('Cartesian basis functions are not supported.')
    if mol.symmetry:
        raise ValueError('Molecular symmetry is not supported.')

    if isinstance(mf, (scf.rohf.ROHF, dft.roks.ROKS)):
        method = 'rohf'
    elif isinstance(mf, (scf.uhf.UHF, dft.uks.UKS)):
        method = 'uhf'
    elif isinstance(mf, (scf.hf.RHF, dft.rks.RKS)):
        method = 'rhf'
    else:
        raise TypeError('Unsupported SCF object %s.' % mf.__class__.__name__)

    def format_config(config, name):
        ''' Validate one spin-resolved `[ns, np, nd, nf]` configuration. '''
        if len(config) != 4:
            raise ValueError('%s must contain [ns, np, nd, nf].' % name)
        if not all(isinstance(n, numbers.Integral) and not isinstance(n, bool)
                   for n in config):
            raise TypeError('%s must contain integers.' % name)
        config = np.asarray(config, dtype=int)
        if np.any(config < 0):
            raise ValueError('%s must contain nonnegative integers.' % name)
        return config

    if method == 'rhf':
        config = format_config(config, 'config')
        if mol.spin != 0:
            raise ValueError('RHF/RKS requires mol.spin = 0.')
        if np.any(config % 2):
            raise ValueError('RHF/RKS requires an even occupation in every channel.')
        if config.sum() != mol.nelectron:
            raise ValueError('config and mol must have the same number of electrons '
                             '(%d != %d).' % (config.sum(), mol.nelectron))
        config_alpha = config_beta = config // 2
    else:
        if len(config) != 2:
            raise ValueError('config must be (config_alpha, config_beta).')
        config_alpha = format_config(config[0], 'config_alpha')
        config_beta = format_config(config[1], 'config_beta')
        nelec = mf.nelec
        if (config_alpha.sum(), config_beta.sum()) != nelec:
            raise ValueError('config and mol must have the same alpha and beta electron '
                             'numbers (%s != %s).' %
                             ((config_alpha.sum(), config_beta.sum()), nelec))

        if method == 'rohf':
            if mol.spin >= 0 and np.any(config_alpha < config_beta):
                raise ValueError('ROHF/ROKS requires config_alpha >= config_beta.')
            if mol.spin < 0 and np.any(config_beta < config_alpha):
                raise ValueError('ROHF/ROKS requires config_beta >= config_alpha.')

    ao_loc = mol.ao_loc_nr()
    ao_l = np.empty(mol.nao, dtype=int)
    for ib in range(mol.nbas):
        ao_l[ao_loc[ib]:ao_loc[ib+1]] = mol.bas_angular(ib)
    ao_idx = {l: np.where(ao_l == l)[0] for l in np.unique(ao_l)}

    for l in range(4):
        nmo = len(ao_idx.get(l, ()))
        if config_alpha[l] > nmo or config_beta[l] > nmo:
            raise ValueError('Not enough l=%d orbitals for config.' % l)

    return method, config_alpha, config_beta, ao_l, ao_idx


def _set_atomic_occ_(mf, method, config_alpha, config_beta, ao_idx, s=None):
    ''' Replace an SCF object's occupation function with l-resolved occupations.

        Args:
            mf (pyscf.scf.hf.SCF):
                PySCF SCF or Kohn-Sham object to modify.
            method (str):
                Occupation type: "rhf", "rohf", or "uhf".
            config_alpha/config_beta (array_like):
                Spin-resolved electron counts by angular momentum.
            ao_idx (dict):
                AO indices grouped by angular momentum.
            s (array_like):
                AO overlap matrix used for dominant-l assignment. Default is None,
                which uses Euclidean coefficient norms.
    '''
    from types import MethodType

    l_values = np.asarray(list(ao_idx))

    def get_mo_l(mo_coeff):
        ''' Assign each molecular orbital to its dominant angular momentum. '''
        if s is None:
            weights = np.asarray([
                np.linalg.norm(mo_coeff[ao_idx[l]], axis=0) for l in l_values
            ])
        else:
            weights = []
            for l in l_values:
                idx = ao_idx[l]
                c_l = mo_coeff[idx]
                s_l = s[np.ix_(idx, idx)]
                weights.append(np.einsum('pi,pq,qi->i',
                                         c_l.conj(), s_l, c_l).real)
            weights = np.asarray(weights)

        imax = np.argmax(weights, axis=0)
        if np.any(weights[imax, np.arange(weights.shape[1])] <= 0):
            raise ValueError('A zero MO was found.')
        return l_values[imax]

    def occupy(mo_energy, mo_coeff, occupations):
        ''' Occupy the lowest-energy orbitals assigned to each angular momentum. '''
        mo_occ = np.zeros_like(mo_energy)
        mo_l = get_mo_l(mo_coeff)
        for l, nocc in enumerate(occupations):
            idx = np.where(mo_l == l)[0]
            if len(idx) < nocc:
                raise ValueError('Not enough MOs assigned to l=%d.' % l)
            idx = idx[np.argsort(mo_energy[idx], kind='stable')]
            mo_occ[idx[:nocc]] = 1
        return mo_occ

    if method == 'rhf':
        def get_occ_(mf, mo_energy=None, mo_coeff=None):
            ''' Return restricted occupations satisfying the requested configuration. '''
            if mo_energy is None:
                mo_energy = mf.mo_energy
            if mo_coeff is None:
                mo_coeff = mf.mo_coeff
            return 2 * occupy(mo_energy, mo_coeff, config_alpha)
    elif method == 'uhf':
        def get_occ_(mf, mo_energy=None, mo_coeff=None):
            ''' Return unrestricted occupations satisfying the requested configuration. '''
            if mo_energy is None:
                mo_energy = mf.mo_energy
            if mo_coeff is None:
                mo_coeff = mf.mo_coeff
            return np.asarray([
                occupy(mo_energy[0], mo_coeff[0], config_alpha),
                occupy(mo_energy[1], mo_coeff[1], config_beta),
            ])
    else:
        def get_occ_(mf, mo_energy=None, mo_coeff=None):
            ''' Return open-shell occupations satisfying the requested configuration. '''
            if mo_energy is None:
                mo_energy = mf.mo_energy
            if mo_coeff is None:
                mo_coeff = mf.mo_coeff
            nclosed = np.minimum(config_alpha, config_beta)
            ntotal = np.maximum(config_alpha, config_beta)
            return (occupy(mo_energy, mo_coeff, nclosed) +
                    occupy(mo_energy, mo_coeff, ntotal))

    mf.get_occ = MethodType(get_occ_, mf)


def atomic_scf_with_pure_l_config_(mf, config):
    ''' Modify an atomic SCF object to preserve a pure-l configuration.

        Args:
            mf (pyscf.scf.hf.SCF):
                PySCF atomic SCF or Kohn-Sham object to modify.
            config (array_like):
                Electron counts `[ns, np, nd, nf]` for RHF/RKS or an `(alpha, beta)`
                pair of such arrays for unrestricted and restricted open-shell methods.

        Return:
            mf (pyscf.scf.hf.SCF):
                Modified SCF object.

        Note:
            The Fock equation is solved independently in each angular-momentum block.
            Orbitals are sorted globally by energy and occupied within each l channel;
            the SCF gradient is projected onto the same block-diagonal space.
    '''
    import inspect
    from types import MethodType

    from pyscf import lib

    method, config_alpha, config_beta, ao_l, ao_idx = \
        _prepare_atomic_config(mf, config)
    mol = mf.mol
    l_values = np.asarray(list(ao_idx))

    same_l = ao_l[:,None] == ao_l[None,:]
    eigh0 = getattr(mf, '_pygto_eigh0', mf._eigh)
    get_veff0 = getattr(mf, '_pygto_get_veff0', mf.get_veff)
    get_grad0 = getattr(mf, '_pygto_get_grad0', mf.get_grad)
    mf._pygto_eigh0 = eigh0
    mf._pygto_get_veff0 = get_veff0
    mf._pygto_get_grad0 = get_grad0

    eigh0_args = inspect.signature(eigh0).parameters

    def _eigh_(mf, f, s, overwrite=False, x=None):
        ''' Solve generalized eigenproblems independently in each l block. '''
        mo_energy = []
        mo_coeff = []
        for l in l_values:
            idx = ao_idx[l]
            f_l = f[np.ix_(idx, idx)]
            s_l = s[np.ix_(idx, idx)]
            kwargs = {}
            if 'overwrite' in eigh0_args:
                kwargs['overwrite'] = overwrite
            if 'x' in eigh0_args:
                kwargs['x'] = None
            e_l, c_l = eigh0(f_l, s_l, **kwargs)

            coeff = np.zeros((mol.nao, len(e_l)), dtype=c_l.dtype)
            coeff[idx] = c_l
            mo_energy.append(e_l)
            mo_coeff.append(coeff)

        mo_energy = np.hstack(mo_energy)
        mo_coeff = np.hstack(mo_coeff)
        order = np.argsort(mo_energy, kind='stable')
        return mo_energy[order], mo_coeff[:,order]

    def project_fock(fock):
        ''' Project a Fock-like array and its array tags onto same-l blocks. '''
        def project_array(a):
            ''' Zero cross-angular-momentum matrix blocks when dimensions match. '''
            a = np.asarray(a)
            if a.ndim >= 2 and a.shape[-2:] == same_l.shape:
                return a * same_l
            return a

        projected = project_array(fock)
        if hasattr(fock, '__dict__'):
            tags = {}
            for key, value in fock.__dict__.items():
                if isinstance(value, np.ndarray):
                    value = project_array(value)
                tags[key] = value
            projected = lib.tag_array(projected, **tags)
        return projected

    def get_veff_(mf, *args, **kwargs):
        ''' Return the effective potential projected onto same-l blocks. '''
        return project_fock(get_veff0(*args, **kwargs))

    def get_grad_(mf, mo_coeff, mo_occ, fock=None):
        ''' Return the SCF gradient using a same-l-projected Fock matrix. '''
        if fock is None:
            dm = mf.make_rdm1(mo_coeff, mo_occ)
            fock = mf.get_fock(dm=dm)
        return get_grad0(mo_coeff, mo_occ, project_fock(fock))

    mf._eigh = MethodType(_eigh_, mf)
    _set_atomic_occ_(mf, method, config_alpha, config_beta, ao_idx)
    mf.get_veff = MethodType(get_veff_, mf)
    mf.get_grad = MethodType(get_grad_, mf)
    return mf


def atomic_scf_with_dominant_l_config_(mf, config):
    ''' Modify an atomic SCF object to preserve a dominant-l configuration.

        Args:
            mf (pyscf.scf.hf.SCF):
                PySCF atomic SCF or Kohn-Sham object to modify.
            config (array_like):
                Electron counts using the convention of
                `atomic_scf_with_pure_l_config_`.

        Return:
            mf (pyscf.scf.hf.SCF):
                Modified SCF object.

        Note:
            The Fock matrix is unchanged. Each generally mixed orbital is assigned to
            the angular momentum with its largest overlap-weighted population, and the
            lowest-energy assigned orbitals are occupied according to `config`.
    '''
    method, config_alpha, config_beta, _, ao_idx = \
        _prepare_atomic_config(mf, config)

    for name, original_name in (
            ('_eigh', '_pygto_eigh0'),
            ('get_veff', '_pygto_get_veff0'),
            ('get_grad', '_pygto_get_grad0')):
        if hasattr(mf, original_name):
            setattr(mf, name, getattr(mf, original_name))

    s = mf.get_ovlp()
    _set_atomic_occ_(mf, method, config_alpha, config_beta, ao_idx, s=s)
    return mf


if __name__ == '__main__':
    from pyscf import gto, scf
    from pygto.basis import BasisSpec

    atm = 'Cu'
    basis = 'cc-pvdz'
    basis = gto.basis.load(basis, atm)
    spec = BasisSpec.init_from_pyscf_basis(basis, atm=atm)
    basis = spec.get_pyscf_basis(keep_l=[0,1,2])

    spin = 1
    # config = [[4, 6, 5, 0], [3, 6, 5, 0]]   # 3d^10 4s^1
    config = [[4, 6, 5, 0], [4, 6, 4, 0]]   # 3d^9 4s^2

    mol = gto.M(atom=atm, basis=basis, spin=spin).set(verbose=4)

    mf = scf.UHF(mol).run()
    eref = mf.e_tot

    mf = scf.UHF(mol)
    atomic_scf_with_pure_l_config_(mf, config)
    mf.kernel()
    ehy = mf.e_tot

    mf = scf.UHF(mol)
    atomic_scf_with_dominant_l_config_(mf, config)
    mf.kernel()
    ehy1 = mf.e_tot

    print(f'{eref: .10f}')
    print(f'{ehy: .10f}')
    print(f'{ehy1: .10f}')
