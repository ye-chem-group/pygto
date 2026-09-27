import numpy as np
from .atomic_scf import atomic_scf_with_pure_l_config_


def get_cost_func(atm, HF, mol_settings=None, CORR=None, corr_settings=None, keep_l=None,
                  keep_channel=None, config=None, basis_fix=None):
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
            keep_channel (int or list of int):
                Basis channels to retain. Default is None, which keeps all.
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
        basis = spec.get_pyscf_basis(keep_l=keep_l, keep_channel=keep_channel)
        if basis_fix is not None:
            basis = basis + basis_fix
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
                         corr=True, corr_settings=None, gamma_vjk=0.1, gamma_mp2=10,
                         auxbasis_fix=None):
    ''' Construct an auxiliary-basis cost function from HF and MP2 errors.

        The reference calculation uses exact two-electron integrals. For each
        candidate auxiliary basis, the returned function measures density-fitting
        errors in the HF Coulomb and exchange matrices and their energy
        contributions, all evaluated with the reference HF density. When
        `corr=True`, it also measures the density-fitted MP2 correlation-energy
        error and the same-spin and opposite-spin error metrics of Weigend et al.,
        J. Chem. Phys. 116, 3175–3183 (2002), Eq. 5.

        The returned function has the signatures

            cost_func(spec) -> cost
            cost_func(spec, full_output=True) -> cost, cost_details

        The unscaled error components are

            max(abs(vj - vj_ref))
            max(abs(vk - vk_ref))
            abs(ej - ej_ref)
            abs(ek - ek_ref)
            abs(emp2 - emp2_ref)
            abs(de_ss / emp2_ss_ref)
            abs(de_os / emp2_os_ref)

        where `vj` and `vk` are the HF Coulomb and exchange matrices, `ej` and
        `ek` are their energy contributions, `emp2` is the MP2 correlation
        energy, and `de_ss` and `de_os` are the same-spin and opposite-spin
        Weigend error metrics, reported as `t2ss err` and `t2os err`,
        respectively. If a reference spin component of the MP2 correlation
        energy is zero, its corresponding normalized error is set to zero. The
        three MP2 components are omitted when `corr=False`.

        The J/K matrix errors are multiplied by `gamma_vjk`, and the normalized
        same-spin and opposite-spin metrics are multiplied by `gamma_mp2`. The
        energy errors are not scaled. The scalar cost is the largest scaled
        component. With `full_output=True`, `cost_details` contains
        `(name, error, scaled_error)` for every component.

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
                Whether to include the three MP2 error components. Default is
                True.
            corr_settings (dict):
                Settings applied to MP2 object through `set`. Default is None.
            gamma_vjk (float):
                Scaling factor for the J/K matrix errors. Default is 0.1.
            gamma_mp2 (float):
                Scaling factor for the normalized same-spin and opposite-spin
                Weigend error metrics. Default is 10.
            auxbasis_fix (pyscf-recognizable basis format):
                Fixed auxiliary functions appended to every candidate basis.
                Default is None.

        Note:
            Relative to the unscaled energy errors, the default `gamma_vjk` places
            a looser requirement on the J/K matrix errors, whereas the default
            `gamma_mp2` places a tighter requirement on the normalized same-spin
            and opposite-spin metrics.

        Return:
            cost_func (callable):
                Function accepting an auxiliary `BasisSpec` and returning its
                maximum scaled error. With `full_output=True`, it also returns
                the component details described above.
    '''

    import inspect
    from pyscf import mp, lib

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

    def get_vjk_ejk(mf):
        vj, vk = mf.get_jk(dm=dm_ref)
        if dm_ref.ndim == 2:    # RHF
            ej = np.einsum('ij,ji->', vj, dm_ref) * 0.5
            ek = np.einsum('ij,ji->', vk, dm_ref) * 0.25
        else:
            vj = vj.sum(axis=0)
            ej = np.einsum('ij,xji->', vj, dm_ref) * 0.5
            ek = np.einsum('xij,xji->', vk, dm_ref) * 0.5
        return vj, vk, ej, ek

    vj_ref, vk_ref, ej_ref, ek_ref = get_vjk_ejk(mf_ref)

    if corr:
        mc_ref = mp.MP2(mf_ref)
        if corr_settings is not None:
            mc_ref.set(**corr_settings)
        eris_ref = mc_ref.ao2mo()
        mc_ref.kernel(eris=eris_ref)
        ecorr_ref = mc_ref.e_corr
        ecorr_ss_ref = mc_ref.e_corr_ss
        ecorr_os_ref = mc_ref.e_corr_os

    def cost_func(spec, full_output=False):
        ''' Evaluate the density-fitting cost for an auxiliary `BasisSpec`.

            Args:
                spec (BasisSpec):
                    Candidate auxiliary-basis specification.
                full_output (bool):
                    Whether to return component-level error details. Default is
                    False.

            Return:
                cost (float):
                    Maximum scaled error component.
                cost_details (list of tuple):
                    Returned only when `full_output=True`. Each tuple contains
                    `(name, error, scaled_error)` for one component.
        '''
        auxbasis = spec.get_pyscf_basis()
        if auxbasis_fix is not None:
            auxbasis = auxbasis + auxbasis_fix
        mf = HF(mol).density_fit(auxbasis)
        if config is not None:
            atomic_scf_with_pure_l_config_(mf, config)
        mf.kernel()

        vj, vk, ej, ek = get_vjk_ejk(mf)

        error_name = ['vj err', 'vk err', 'ej err', 'ek err']
        error_vector = np.asarray((
            abs(vj-vj_ref).max(),
            abs(vk-vk_ref).max(),
            abs(ej-ej_ref),
            abs(ek-ek_ref),
        ))
        scaled_error_vector = error_vector.copy()
        scaled_error_vector[:2] *= gamma_vjk

        if corr:
            # Ref: J. Chem. Phys. 116, 3175 (2002)
            mc = mp.MP2(mf_ref).density_fit(auxbasis=auxbasis)
            if corr_settings is not None:
                mc.set(**corr_settings)
            eris = mc.ao2mo()
            mc.kernel(eris=eris)
            ecorr = mc.e_corr

            if hasattr(eris_ref, 'OVOV'):   # UHF
                moe = mc.split_mo_energy()

                ovL, OVL = eris.ovL
                dovov_list = (
                    np.dot(ovL, ovL.T) - eris_ref.ovov,
                    np.dot(ovL, OVL.T) - eris_ref.ovOV,
                    np.dot(OVL, OVL.T) - eris_ref.OVOV,
                )
                # same spin
                de_ss = 0
                for s in [0,1]:
                    s_ovov = 0 if s == 0 else 2
                    nocc = eris.nocc[s]
                    nvir = eris.nvir[s]
                    dovov = dovov_list[s_ovov].reshape(nocc,nvir,nocc,nvir)
                    moe_occ, moe_vir = moe[s][1:3]
                    eia = (moe_occ[:,None] - moe_vir).reshape(-1)
                    eiajb = (eia[:,None] + eia).reshape(*dovov.shape)
                    de = lib.einsum('iajb,iajb->', abs(dovov-dovov.transpose(0,3,2,1)),
                                    abs(dovov)/eiajb) * 0.5
                    de_ss += de

                # oppo spin
                nocca, noccb = eris.nocc
                nvira, nvirb = eris.nvir
                dovov = dovov_list[1].reshape(nocca,nvira,noccb,nvirb)
                moea_occ, moea_vir = moe[0][1:3]
                moeb_occ, moeb_vir = moe[1][1:3]
                eiaa = (moea_occ[:,None] - moea_vir).reshape(-1)
                eiab = (moeb_occ[:,None] - moeb_vir).reshape(-1)
                eiajb = (eiaa[:,None] + eiab).reshape(*dovov.shape)
                de_os = lib.einsum('iajb,iajb->', abs(dovov), abs(dovov)/eiajb)
            else:   # RHF
                nocc, nvir = eris.nocc, eris.nvir
                dovov = (np.dot(eris.ovL, eris.ovL.T) - eris_ref.ovov).reshape(nocc,nvir,nocc,nvir)
                moe_occ, moe_vir = mc.split_mo_energy()[1:3]
                eia = (moe_occ[:,None] - moe_vir).reshape(-1)
                eiajb = (eia[:,None] + eia).reshape(*dovov.shape)
                t2 = dovov/eiajb
                de_ss = lib.einsum('iajb,iajb->', abs(dovov-dovov.transpose(0,3,2,1)), abs(t2))
                de_os = lib.einsum('iajb,iajb->', abs(dovov), abs(t2))

            error_name_corr = ['emp2 err', 't2ss err', 't2os err']
            error_vector_corr = np.asarray((
                abs(ecorr-ecorr_ref),
                abs(de_ss/ecorr_ss_ref) if abs(ecorr_ss_ref) > 1e-10 else 0,
                abs(de_os/ecorr_os_ref) if abs(ecorr_os_ref) > 1e-10 else 0,
            ))
            scaled_error_vector_corr = error_vector_corr.copy()
            scaled_error_vector_corr[-2:] *= gamma_mp2

            error_name += error_name_corr
            error_vector = np.hstack((error_vector, error_vector_corr))
            scaled_error_vector = np.hstack((scaled_error_vector, scaled_error_vector_corr))

        error = max(scaled_error_vector)

        if full_output:
            error_details = [(name,err,serr) for name,err,serr in
                             zip(error_name,error_vector,scaled_error_vector)]
            return error, error_details
        else:
            return error

    return cost_func


if __name__ == '__main__':
    pass
