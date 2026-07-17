import numpy as np
from .atomic_scf import atomic_scf_with_pure_l_config_


def get_cost_func(atm, HF, mol_settings=None, CORR=None, corr_settings=None, keep_l=None,
                  config=None):
    ''' Construct an atomic electronic-structure cost function.

        Args:
            atm (str):
                Atomic symbol.
            HF (class or callable):
                Callable constructing a PySCF SCF object from a molecule.
            mol_settings (dict):
                Molecule attributes applied before building. Default is None.
            CORR (class or callable):
                Callable constructing a correlated method from the SCF object. Default
                is None, which uses the SCF total energy.
            corr_settings (dict):
                Correlated-method attributes applied before execution. Default is None.
            keep_l (int or list of int):
                Basis angular momenta to retain. Default is None, which keeps all.
            config (array_like):
                Pure-angular-momentum electron configuration. Default is None.

        Return:
            cost_func (callable):
                Function accepting a BasisSpec and returning the SCF total energy or
                correlated energy contribution. With `full_output=True`, it also
                returns the completed PySCF method object.
    '''

    import inspect

    if not (inspect.isclass(HF) or callable(HF)):
        raise TypeError('HF must be a class or callable.')

    if CORR is not None:
        if not (inspect.isclass(CORR) or callable(CORR)):
            raise TypeError('CORR must be a class or callable.')

    def get_mol(basis):
        ''' Build a PySCF molecule for basis data. '''
        from pyscf import gto
        mol = gto.Mole()
        mol.atom = atm
        mol.basis = basis
        mol.verbose = 0 # may be overwritten by `mol_settings`
        if mol_settings is not None:
            mol.set(**mol_settings)
        mol.build()
        return mol

    def cost_func(spec, full_output=False):
        ''' Evaluate the electronic-structure cost for a BasisSpec. '''
        basis = spec.get_pyscf_basis(keep_l=keep_l)
        mol = get_mol(basis)
        mf = HF(mol)
        if config is not None:
            atomic_scf_with_pure_l_config_(mf, config)
        mf.kernel()

        if CORR is None:
            e = mf.e_tot
            obj = mf
        else:
            mc = CORR(mf)
            if corr_settings is not None:
                mc.set(**corr_settings)
            mc.kernel()
            e = mc.e_corr
            obj = mc

        if full_output:
            return e, obj
        else:
            return e

    return cost_func


def get_cost_func_auxopt(atm, aobasis, HF, mol_settings=None, config=None,
                         corr=True, corr_settings=None, gamma_vjk=0.1):
    ''' Construct a cost function for auxiliary-basis optimization.

        The returned `:func:cost_func` has the following signature:
            cost_func(spec) -> error
            cost_func(spec, True) -> error, error_vec
        where `error` is a single float to be minimized, while `error_vec`
        includes all error components.

        The error vector is defined as:
            error_vec = (
                abs(vj-vj_ref).max(),
                abs(vk-vk_ref).max(),
                abs(ej-ej_ref),
                abs(ek-ek_ref),
                abs(ecorr-ecorr_ref),
                abs(t2-t2_ref).max()
            )
        where `vj/k` are the HF J/K matrices, `ej/k` are the HF J/K energies,
        and `ecorr` and `t2` are the MP2 correlation energy and T2 amplitudes.
        The MP2 component is only included when `corr=True`.

        The error is defined as:
            error = max(error_vec)

        Args:
            atm (str):
                Atomic symbol.
            aobasis (pyscf-recognizable basis format):
                Orbital basis for which the auxiliary basis is optimized.
            HF (class or callable):
                HF(mol) -> mf
            mol_settings (dict):
                Settings for `mol` through `mol.set(**mol_settings)`. Default is None.
            config (array_like):
                Pure-angular-momentum electron configuration. Default is None.
            corr (bool):
                Whether to include MP2 correlation in the error vector. Default is
                True.
            corr_settings (dict):
                Settings applied to MP2 object through `set`. Default is None.
            gamma_vjk (float):
                Scaling factor for the J/K matrix error. Default is 0.1.

        Return:
            cost_func (callable):
                Function accepting an auxiliary BasisSpec and returning its maximum
                scaled error. With `full_output=True`, it also returns the error vector.
    '''

    import inspect
    from pyscf import mp

    if not (inspect.isclass(HF) or callable(HF)):
        raise TypeError('HF must be a class or callable.')

    def get_mol(basis):
        ''' Build a PySCF molecule for orbital basis data. '''
        from pyscf import gto
        mol = gto.Mole()
        mol.atom = atm
        mol.basis = basis
        mol.verbose = 0 # may be overwritten by `mol_settings`
        if mol_settings is not None:
            mol.set(**mol_settings)
        mol.build()
        return mol

    # get reference
    mol = get_mol(aobasis)
    mf_ref = HF(mol)
    if config is not None:
        atomic_scf_with_pure_l_config_(mf_ref, config)
    mf_ref.kernel()
    dm_ref = mf_ref.make_rdm1()
    nao = dm_ref.shape[-1]
    dm_ref = dm_ref.reshape(-1, nao, nao)
    vj_ref, vk_ref = mf_ref.get_jk(dm=dm_ref)
    ej_ref = np.einsum("xij,xji->", vj_ref, dm_ref)
    ek_ref = np.einsum("xij,xji->", vk_ref, dm_ref) * 0.5

    if corr:
        mc_ref = mp.MP2(mf_ref)
        if corr_settings is not None:
            mc_ref.set(**corr_settings)
        mc_ref.kernel()
        ecorr_ref = mc_ref.e_corr
        t2_ref = mc_ref.t2

    def cost_func(spec, full_output=False):
        ''' Evaluate density-fitting errors for an auxiliary BasisSpec. '''
        auxbasis = spec.get_pyscf_basis()
        mf = HF(mol).density_fit(auxbasis)
        if config is not None:
            atomic_scf_with_pure_l_config_(mf, config)
        mf.kernel()

        vj, vk = mf.get_jk(dm=dm_ref)
        ej = np.einsum("xij,xji->", vj, dm_ref)
        ek = np.einsum("xij,xji->", vk, dm_ref) * 0.5

        error_vector = np.asarray((
            abs(vj-vj_ref).max(),
            abs(vk-vk_ref).max(),
            abs(ej-ej_ref),
            abs(ek-ek_ref),
        ))
        scaled_error_vector = error_vector.copy()
        scaled_error_vector[:2] *= gamma_vjk

        if corr:
            mc = mp.MP2(mf_ref).density_fit(auxbasis=auxbasis)
            if corr_settings is not None:
                mc.set(**corr_settings)
            mc.kernel()
            ecorr = mc.e_corr
            t2 = mc.t2

            if isinstance(t2, np.ndarray):
                error_vector_corr = np.asarray((
                    abs(ecorr-ecorr_ref),
                    abs(t2-t2_ref).max()
                ))
            else:
                error_vector_corr = np.asarray((
                    abs(ecorr-ecorr_ref),
                    np.asarray([abs(t-tr).max() for t,tr in zip(t2,t2_ref)]).max()
                ))

            error_vector = np.hstack((error_vector, error_vector_corr))
            scaled_error_vector = np.hstack((scaled_error_vector, error_vector_corr))

        error = max(scaled_error_vector)

        if full_output:
            return error, scaled_error_vector
        else:
            return error

    return cost_func


if __name__ == '__main__':
    pass
