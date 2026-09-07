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
    kappa0s = [1e10,3e9,1e9,3e8,1e8,3e7,1e7]
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

    results = []
    for kappa0 in kappa0s:
        lindep_penalty_func = lib.pyscf_helper.get_lindep_penalty_func(atm, cell, kappa0)

        ''' Perform Material Constrained Atomic Optimization (MCAO).
            Because MCAO changes ``spec`` in place, next kappa0 will automatically use
            optimized ``spec`` from previous kappa0.
        '''
        opt = MCAO(spec, stages, lindep_penalty_func).set(verbose=5)
        opt.kernel()
        results.append((kappa0, spec.copy()))

    ''' Compare atomic accuracy and solid-state numerical stability with the reference
        cc-pVTZ basis.

        Reference output:

**** Atomic Accuracy and Solid-state Stability ****
Init cc-pVTZ basis:
  ehf= -37.6866622379  eccsd= -0.0933596761  penalty= 1.643e+01  cond= 2.109e+09

MCAO cc-pVTZ basis:
  kappa0= 1e+10  ehf= -37.6866623658  eccsd= -0.0933652758  penalty= 1.503e+01  cond= 1.872e+09
  kappa0= 1e+10  err= -0.0000001279         -0.0000055997

  kappa0= 3e+09  ehf= -37.6866541357  eccsd= -0.0933499791  penalty= 1.170e+01  cond= 1.022e+09
  kappa0= 3e+09  err=  0.0000081022          0.0000096970

  kappa0= 1e+09  ehf= -37.6866085105  eccsd= -0.0933174217  penalty= 7.375e+00  cond= 5.012e+08
  kappa0= 1e+09  err=  0.0000537274          0.0000422544

  kappa0= 3e+08  ehf= -37.6864755187  eccsd= -0.0932502812  penalty= 3.417e+00  cond= 1.986e+08
  kappa0= 3e+08  err=  0.0001867192          0.0001093950

  kappa0= 1e+08  ehf= -37.6862407455  eccsd= -0.0931492355  penalty= 1.260e+00  cond= 7.881e+07
  kappa0= 1e+08  err=  0.0004214924          0.0002104406

  kappa0= 3e+07  ehf= -37.6858284203  eccsd= -0.0929756984  penalty= 2.451e-01  cond= 2.735e+07
  kappa0= 3e+07  err=  0.0008338175          0.0003839777

  kappa0= 1e+07  ehf= -37.6852736864  eccsd= -0.0926247741  penalty= 2.271e-02  cond= 1.007e+07
  kappa0= 1e+07  err=  0.0013885515          0.0007349020

        As kappa0 is tightened from 1e10 (nearly no penalty) to 1e7 (strong penalty), both
        the atomic HF and CCSD correlation energy errors increase to about 1 mEh, but the
        condition number decreases from 2e9 to 1e7. Also, starting from kappa0 = 1e9 and
        downwards, we see that the actual condition number closely follows kappa0.
    '''
    spec.log_note('**** Atomic Accuracy and Solid-state Stability ****')

    ehf_init = cost_func_hf(spec_init)
    ecorr_init = cost_func_ccsd(spec_init)
    penalty_init, cond_init = lindep_penalty_func(spec_init)
    spec.log_note('Init cc-pVTZ basis:')
    spec.log_note('ehf= %13.10f  eccsd= %13.10f  penalty= %.3e  cond= %.3e' % (
        ehf_init, ecorr_init, penalty_init, cond_init), indent=1)
    spec.log_note('')

    spec.log_note('MCAO cc-pVTZ basis:')
    for kappa0,spec in results:
        ehf = cost_func_hf(spec)
        ecorr = cost_func_ccsd(spec)
        penalty, cond = lindep_penalty_func(spec)
        spec.log_note('kappa0= %.0e  ehf= %13.10f  eccsd= %13.10f  penalty= %.3e  cond= %.3e' % (
            kappa0, ehf, ecorr, penalty, cond), indent=1)
        spec.log_note('kappa0= %.0e  err= %13.10f         %13.10f' % (
            kappa0, ehf-ehf_init, ecorr-ecorr_init), indent=1)
        spec.log_note('')

    spec.log_note('Initial cc-pVTZ basis:')
    spec_init.dump_basis()
    spec.log_note('')
    for kappa0,spec in results:
        spec.log_note('MCAO-cc-pVTZ basis (with kappa0= %.3e):' % kappa0)
        spec.dump_basis()
