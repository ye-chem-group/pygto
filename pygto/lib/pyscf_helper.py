''' Helper functions for PySCF users.

    To ensure the PyGTO package works for non-PySCF users, the `pyscf_helper` module is
    not imported in `__init__.py`. Therefore, PySCF users should do the following
    ```
        from pygto.lib.pyscf_helper import [function]
    ```
    to import functions from this module.
'''


def get_cost_func(atm, HF, mol_settings=None, CORR=None, corr_settings=None, keep_l=None):

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

    def cost_func(spec, full_output=False):
        basis = spec.get_pyscf_basis(keep_l=keep_l)
        mol = get_mol(basis)
        mf = HF(mol)
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
