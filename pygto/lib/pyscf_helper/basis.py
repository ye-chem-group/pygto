from ..numeric_helper import to_int_list


def load_basis(fbas, atm, unc=False, keep_l=None):
    ''' Load basis data from named basis or basis file.

        Args:
            fbas (str):
                Named basis (e.g., cc-pvdz, def2-tzvp) or path to a basis file.
            atm (str):
                Atomic symbol.
            unc (bool):
                Whether to decontract the basis. Default is False.
            keep_l (int or list of int):
                Angular momenta to keep. Default is None, which keeps all channels.

        Return:
            basis (list):
                PySCF-format basis, i.e.,
                [
                    (l1, (e1, c11, c12), (e2, c21, c22), ...),
                    (l2, ...),
                    ...
                ]
    '''
    from pyscf import gto
    basis = gto.basis.load(fbas, atm)

    if unc:
        basis = gto.uncontracted_basis(basis)

    if keep_l is not None:
        keep_l = to_int_list(keep_l)
        basis = [b for b in basis if int(b[0]) in keep_l]

    return basis
