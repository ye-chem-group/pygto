''' This example demonstrates how to use the MCAO workflow to generate basis sets suitable
    for solid-state calculations.

    MCAO minimizes the following cost function:

        E_stage(alpha) + penalty_strength * penalty_rescale
                       * penalty(alpha; kappa0, structures)

    Here, `alpha` denotes the Gaussian exponents, and `penalty` is the
    linear-dependence penalty (LDP) evaluated for a set of reference structures.
    Approximately, `kappa0` is the target condition number: relative overlap
    eigenvalues below about `1/kappa0` are penalized. The parameter
    `penalty_strength` sets the overall energy scale of the penalty, while
    `penalty_rescale` adjusts its contribution in each optimization stage.

    In practice, one can fix `penalty_strength` at a reasonable energy scale (0.01 Ha
    is a useful starting value) and scan `kappa0` to generate MCAO basis sets with
    different degrees of linear dependence. A larger `kappa0` imposes a weaker
    constraint and yields a basis closer to the atom-optimized result. A smaller
    `kappa0` imposes a stronger constraint, typically at the cost of reduced atomic
    accuracy.

    In this example, we generate an MCAO basis for carbon with `kappa0 = 1e8`. We start
    from cc-pVTZ and use diamond at its experimental lattice constant (3.567 Å) as the
    reference structure.

    NOTE: This example relies on PySCF for atomic ROHF and CCSD calculations and for
    periodic AO overlap matrices.

    NOTE: The MCAO calculation in this example may take 5--10 minutes.
'''

from pathlib import Path

from pygto import lib
from pygto.basis import BasisSpec
from pygto.workflow import MCAO
from pygto.data.elements import get_spin

from pyscf import scf, cc


DATA_DIR = Path(__file__).resolve().parent / 'data'


if __name__ == '__main__':
    atm = 'C'
    spin = get_spin(atm)
    basis = 'cc-pvtz'
    val_l = [0, 1]
    pol_l = [2, 3]
    frozen = 1
    # In practice, scan kappa0 over values such as 1e10, 3e9, ..., 1e7.
    kappa0 = 1e8
    # Structure used to evaluate the linear-dependence penalty.
    fvasp = str(DATA_DIR / 'diamond.vasp')

    ''' Construct the ROHF total-energy and CCSD correlation-energy cost functions.
    '''
    cost_func_hf = lib.pyscf_helper.get_cost_func(
        atm, scf.ROHF, mol_settings={'spin':spin}, keep_l=val_l,
    )
    cost_func_ccsd = lib.pyscf_helper.get_cost_func(
        atm, scf.ROHF, mol_settings={'spin':spin},
        CORR=cc.CCSD, corr_settings={'frozen':frozen},
    )

    ''' Initialize BasisSpec from a named basis.
    '''
    spec = BasisSpec.init_from_basis(basis, atm)
    spec_init = spec.copy()

    ''' Construct the `stages` for MCAO

        Stage 1: HF energy optimization of the valence set
        Stage 2: CCSD correlation energy optimization of the polarization set

        Each stage is a `dict` that defines `prefix`, `cost_func`, and other optional settings.
    '''
    stages = [
        {
            'prefix': 'ehf',
            'cost_func': cost_func_hf,
            'penalty_rescale': 1.,  # default
            'active_l': val_l,      # only optimize valence shells
        },
        {
            'prefix': 'ecorr',
            'cost_func': cost_func_ccsd,
            # Scale the penalty contribution by 0.1 in the correlation stage.
            'penalty_rescale': 0.1,
            'active_l': pol_l,      # only optimize polarization shells
        },
    ]

    ''' Construct the linear-dependence penalty function.

        For customized LDP, ensure the following functional signature:

            def lindep_penalty_func(spec, scale=1.):
                # your implementation
                return penalty, cond

        where `scale` is the rigid scaling factor applied to the reference solids,
        `penalty` is the LDP, and `cond` is the condition number.
    '''
    lat = lib.Lattice.init_from_vasp_poscar(fvasp)
    cell = lat.get_pyscf_cell()
    lindep_penalty_func = lib.pyscf_helper.get_lindep_penalty_func(atm, cell, kappa0)

    ''' Perform Material Constrained Atomic Optimization (MCAO).
    '''
    opt = MCAO(spec, stages, lindep_penalty_func).set(verbose=5)
    opt.kernel()

    ''' Compare atomic accuracy and solid-state numerical stability with the reference
        cc-pVTZ basis.

        Reference output:

            **** Atomic Accuracy and Solid-state Stability ****
            Init cc-pVTZ basis: ehf= -37.6866622379  eccsd= -0.0933596761
                                penalty= 9.478e-01  cond= 2.109e+09
            MCAO-cc-pVTZ basis: ehf= -37.6859315498  eccsd= -0.0929533574
                                penalty= 2.460e-02  cond= 3.458e+07

        Both the atomic HF and CCSD correlation energies are slightly less accurate, but
        the MCAO basis has improved numerical stability, as indicated by its lower LDP
        and condition number.
    '''
    spec.log_note('**** Atomic Accuracy and Solid-state Stability ****')

    ehf = cost_func_hf(spec_init)
    ecorr = cost_func_ccsd(spec_init)
    penalty, cond = lindep_penalty_func(spec_init)
    spec.log_note('Init cc-pVTZ basis: ehf= %.10f  eccsd= %.10f' % (ehf, ecorr))
    spec.log_note('                    penalty= %.3e  cond= %.3e' % (penalty, cond))

    ehf = cost_func_hf(spec)
    ecorr = cost_func_ccsd(spec)
    penalty, cond = lindep_penalty_func(spec)
    spec.log_note('MCAO-cc-pVTZ basis: ehf= %.10f  eccsd= %.10f' % (ehf, ecorr))
    spec.log_note('                    penalty= %.3e  cond= %.3e' % (penalty, cond))
    spec.log_note('')

    spec.log_note('Initial cc-pVTZ basis:')
    spec_init.dump_basis()
    spec.log_note('')
    spec.log_note('MCAO-cc-pVTZ basis (with kappa0= %.3e):' % kappa0)
    spec.dump_basis()
