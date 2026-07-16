import numpy as np
from .atomic_scf import atomic_scf_with_pure_l_config_


def get_ano_input_func(atm, HF, mol_settings=None, config=None, CORR=None, corr_settings=None):
    ''' Calculates spin-summed density matrix and AO overlap matrix.

        Note:
            The order of the AO basis functions in which the two matrices are calculated
            must be consistent with the order in `spec.get_basis_str_nwchem()`. The order
            of the m-sublevels do not matter. For example, using (px, py, pz) or (pz, py, px)
            for the l=1 channel does not affect the final results.
    '''
    import inspect

    if not (inspect.isclass(HF) or callable(HF)):
        raise TypeError('HF must be a class or callable.')

    if CORR is not None:
        if not (inspect.isclass(CORR) or callable(CORR)):
            raise TypeError('CORR must be a class or callable.')

    def get_mol(basis):
        from pyscf import gto
        mol = gto.Mole()
        mol.atom = atm
        mol.basis = basis
        mol.verbose = 0 # may be overwritten by `mol_settings`
        if mol_settings is not None:
            mol.set(**mol_settings)
        mol.build()
        return mol

    def ano_input_func(spec):
        basis = spec.get_pyscf_basis()
        mol = get_mol(basis)
        mf = HF(mol)
        if config is not None:
            atomic_scf_with_pure_l_config_(mf, config)
        mf.kernel()

        if not mf.converged:
            raise RuntimeError('SCF not converged.')

        if CORR is not None:
            mc = CORR(mf)
            if corr_settings is not None:
                mc.set(**corr_settings)
            mc.kernel()
            dm = mc.make_rdm1(ao_repr=True)
        else:
            dm = mf.make_rdm1()

        nao = mol.nao_nr()
        dm = np.asarray(dm).reshape(-1,nao,nao)
        dm = np.sum(dm, axis=0)

        s = mf.get_ovlp()
        return dm, s

    return ano_input_func
