import numpy as np

from ..lattice_helper import Lattice


__all__ = ['get_lindep_penalty_func']


def get_lindep_penalty_func(atm, cell, kappa0, natm_min=300, ev_min=1e-8, sigmoid_p=2,
                            keep_l=None, verbose=None):
    ''' Construct a periodic overlap linear-dependence penalty function.

        Args:
            atm (str):
                Atomic symbol whose basis is replaced.
            cell (pyscf.pbc.gto.Cell):
                Reference periodic cell.
            kappa0 (float):
                Target overlap condition number.
            natm_min (int):
                Minimum effective atom count controlling k-point meshes. Default is 300.
            ev_min (float):
                Smooth lower bound for overlap eigenvalues. Default is `1e-8`.
            sigmoid_p (float):
                Exponent controlling penalty sharpness. Default is 2.
            keep_l (int or list of int):
                Angular momenta to keep. Default is None, which keeps all channels.
            verbose (int):
                Logging verbosity. Default is None.

        Return:
            get_lindep_penalty (callable):
                Function accepting `(spec, scale=1.)` and returning the penalty and
                maximum overlap condition number.
    '''
    lattice = Lattice.init_from_pyscf_cell(cell)
    if verbose is not None: lattice.verbose = verbose
    scaled_kpts, kpts_deg = get_uniq_kpts(cell, natm_min, verbose=verbose)
    lattice.log_note('Using %d kpts for lindep penalty' % (len(scaled_kpts)))
    lattice.log_info('')
    lattice.log_info('%-26s  %s' % ('scaled_kpts', 'deg'))
    for k,d in zip(scaled_kpts,kpts_deg):
        lattice.log_info('% 7.5f % 7.5f % 7.5f  %3d' % (*k,d))
    lattice.log_info('')

    basis_full = dict(cell._basis)

    def lindep_penalty_func(spec, scale=1.):
        ''' Evaluate periodic overlap penalty and condition number. '''
        basis_full[atm] = spec.get_pyscf_basis(keep_l=keep_l)
        cell = lattice.get_pyscf_cell(basis=basis_full, scale=scale,
                                      cell_settings={'precision':1e-12})
        kpts = cell.get_abs_kpts(scaled_kpts)
        ks1e = cell.pbc_intor('int1e_ovlp', kpts=kpts)

        ek = [np.linalg.eigvalsh(s) for s in ks1e]
        emax = np.max([np.max(e) for e in ek])
        e0 = emax / kappa0

        emin = np.min([np.min(abs(e)) for e in ek])
        cond = emax/emin

        penalty = np.sum([
            d*np.sum(sigmoid(e/e0, sigmoid_p, lower_bound=ev_min))
            for e,d in zip(ek,kpts_deg)
        ])
        penalty /= np.sum(kpts_deg)

        return penalty, cond

    return lindep_penalty_func


def get_uniq_kpts(cell, natm_min=300, verbose=None):
    ''' Determine a union of unique k-points and their degeneracies.

        Args:
            cell (pyscf.pbc.gto.Cell):
                Reference periodic cell.
            natm_min (int):
                Minimum effective atom count controlling the largest k mesh. Default
                is 300.
            verbose (int):
                Logging verbosity. Default is None.

        Return:
            scaled_kpts (ndarray):
                Unique scaled k-points.
            degeneracies (ndarray):
                Number of occurrences of each k-point across sampled meshes.
    '''
    # TODO: better handle of noncubic crystals
    from pyscf.pbc.lib.kpts_helper import unique

    lattice = Lattice.init_from_pyscf_cell(cell)
    if verbose is not None: lattice.set(verbose=verbose)

    symmetry = True
    try:
        cell = lattice.get_pyscf_cell(symmetry=True)
    except:
        cell = lattice.get_pyscf_cell(symmetry=False)
        symmetry = False

    if not symmetry:
        lattice.log_warn('Symmetry is not used for lindep penalty')

    nkmax = np.ceil((float(natm_min)/cell.natm)**(1./3.)).astype(int)
    lattice.log_info('Generating kpts for lindep penalty with nkmax= %d' % (nkmax))

    kpts_union = []
    for nk in range(1,nkmax+1):
        kmesh = [nk]*3
        if symmetry:
            kpts = cell.make_kpts(kmesh, space_group_symmetry=True,
                                  time_reversal_symmetry=True)
            kpts_union.append(kpts.kpts_ibz)
        else:
            kpts = cell.make_kpts(kmesh)
            kpts_union.append(kpts)
    kpts, _, uniq_inv = unique(np.vstack(kpts_union))
    nkpts = len(kpts)
    kpts_deg = np.asarray([np.count_nonzero(uniq_inv==k) for k in range(nkpts)]).astype(float)
    scaled_kpts = cell.get_scaled_kpts(kpts)

    return scaled_kpts, kpts_deg


def safe_zero(x, lower_bound):
    ''' Smoothly bound the magnitude of a value away from zero.

        Args:
            x (array_like):
                Input values.
            lower_bound (float):
                Positive smoothing bound.

        Return:
            value (scalar or ndarray):
                `sqrt(x**2 + lower_bound**2)`.
    '''
    return (x**2 + lower_bound**2)**0.5


def sigmoid(x, p, lower_bound=1e-10):
    ''' Evaluate the decreasing penalty sigmoid `1 / (1 + x**p)`.

        Args:
            x (array_like):
                Input values.
            p (float):
                Sigmoid exponent.
            lower_bound (float):
                Smooth lower magnitude bound. Default is `1e-10`.

        Return:
            value (scalar or ndarray):
                Sigmoid values.
    '''
    x = safe_zero(x, lower_bound)
    return 1./(1. + np.power(x, p))
