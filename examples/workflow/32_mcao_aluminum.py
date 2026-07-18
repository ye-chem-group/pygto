''' This example demonstrates how to use the MCAO workflow to generate basis sets suitable
    for solid-state calculations.

    For a short introduction to MCAO, see the module docstring in `31_mcao_carbon.py`.

    In this example, we generate an MCAO basis for aluminum with `kappa0 = 1e8`. The
    initial basis combines the TCAO-optimized valence channels from
    `12_tcao_aluminum.py` with the polarization channel of cc-pVDZ. Both fcc aluminum
    and aluminum nitride (AlN) are used as reference solids.

    Compared with `31_mcao_carbon.py`, this example additionally demonstrates how to:

        1. Supply fixed bases for elements other than the element being optimized, such
           as nitrogen in AlN.
        2. Combine linear-dependence penalties from multiple reference solids.

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
    atm = 'Al'
    spin = get_spin(atm)
    # Take the initial valence set from `12_tcao_aluminum.py`.
    basis_val = str(DATA_DIR / 'al_gth-hf-rev_val.dat')
    val_l = [0, 1]
    # Take the initial polarization set from all-electron cc-pVDZ.
    basis_pol = 'cc-pvdz'
    pol_l = [2]
    pseudo = 'gth-hf-rev'
    frozen = 0
    # In practice, scan kappa0 over values such as 1e10, 3e9, ..., 1e7.
    kappa0 = 1e8
    # Structures used to evaluate the linear-dependence penalty.
    reference_solids = [
        # Label, VASP POSCAR path.
        ('Al', str(DATA_DIR / 'aluminum.vasp')),
        ('AlN', str(DATA_DIR / 'aluminum_nitride.vasp')),
    ]
    # Supply fixed bases for elements other than Al. Here, nitrogen uses GTH-cc-pVDZ.
    basis_extra = {
        'N': str(DATA_DIR / 'n_gth-cc-pvdz.dat')
    }

    ''' Construct the ROHF total-energy and CCSD correlation-energy cost functions.
    '''
    cost_func_hf = lib.pyscf_helper.get_cost_func(
        atm, scf.ROHF, mol_settings={'spin':spin, 'pseudo':pseudo}, keep_l=val_l,
    )
    cost_func_ccsd = lib.pyscf_helper.get_cost_func(
        atm, scf.ROHF, mol_settings={'spin':spin, 'pseudo':pseudo},
        CORR=cc.CCSD, corr_settings={'frozen':frozen},
    )

    ''' Construct the initial BasisSpec from the file-based valence basis and the
        named polarization basis.
    '''
    spec_val = BasisSpec.init_from_basis(basis_val, atm, keep_l=val_l)
    spec_pol = BasisSpec.init_from_basis(basis_pol, atm, keep_l=pol_l)
    spec = spec_val.merge(spec_pol)
    spec_init = spec.copy()

    ''' Construct the optimization stages for MCAO.

        Stage 1: HF energy optimization of the valence set
        Stage 2: CCSD correlation energy optimization of the polarization set

        Each stage is a dictionary defining `prefix`, `cost_func`, and optional settings.
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

    ''' Construct a linear-dependence penalty function for multiple solids.

        For customized LDP, ensure the following functional signature:

            def lindep_penalty_func(spec, scale=1.):
                # your implementation
                return penalty, cond

        Here, `scale` uniformly scales the reference solids, `penalty` is the summed
        LDP over all reference solids, and `cond` is the largest condition number among
        them.
    '''
    # The Al entry is a placeholder that the penalty function replaces with `spec`.
    basis_full = {atm: 'def2-svp'}
    basis_full.update(basis_extra)
    lindep_penalty_funcs = []
    for fml,fvasp in reference_solids:
        lat = lib.Lattice.init_from_vasp_poscar(fvasp)
        cell = lat.get_pyscf_cell(basis=basis_full)
        lindep_penalty_func = lib.pyscf_helper.get_lindep_penalty_func(atm, cell, kappa0)
        lindep_penalty_funcs.append( lindep_penalty_func )

    def lindep_penalty_func(spec, scale=1.):
        penalty = cond = 0.
        for func in lindep_penalty_funcs:
            penalty1, cond1 = func(spec, scale)
            penalty += penalty1
            cond = max(cond, cond1)
        return penalty, cond

    ''' Perform Material Constrained Atomic Optimization (MCAO).
    '''
    opt = MCAO(spec, stages, lindep_penalty_func).set(verbose=5)
    opt.kernel()

    ''' Compare the atomic accuracy and solid-state numerical stability of the initial
        and MCAO-optimized bases.

        Reference output:

            **** Atomic Accuracy and Solid-state Stability ****
            Init cc-pVDZ basis: ehf= -1.8836488740  eccsd= -0.0527646079
                                penalty= 4.509e+00  cond= 4.734e+14  (overall)
                                penalty= 2.287e+00  cond= 4.734e+14  (Al)
                                penalty= 2.222e+00  cond= 3.535e+12  (AlN)
            MCAO-cc-pVDZ basis: ehf= -1.8832348522  eccsd= -0.0525542361
                                penalty= 8.854e-03  cond= 5.935e+07  (overall)
                                penalty= 8.381e-03  cond= 5.935e+07  (Al)
                                penalty= 4.736e-04  cond= 3.497e+06  (AlN)

        Both the atomic HF and CCSD correlation energies are slightly less accurate, but
        the MCAO basis is substantially more stable numerically, as indicated by its
        lower LDP and condition number. For example, the condition number of Al decreases
        from about 4e14 to 6e7, while that of AlN decreases from about 4e12 to 3e6.
    '''
    spec.log_note('**** Atomic Accuracy and Solid-state Stability ****')

    ehf = cost_func_hf(spec_init)
    ecorr = cost_func_ccsd(spec_init)
    penalty, cond = lindep_penalty_func(spec_init)
    spec.log_note('Init cc-pVDZ basis: ehf= %.10f  eccsd= %.10f' % (ehf, ecorr))
    spec.log_note('                    penalty= %.3e  cond= %.3e  (overall)' % (penalty, cond))
    for ldp_func,(fml,fvasp) in zip(lindep_penalty_funcs,reference_solids):
        penalty, cond = ldp_func(spec_init)
        spec.log_note('                    penalty= %.3e  cond= %.3e  (%s)' % (penalty, cond, fml))

    ehf = cost_func_hf(spec)
    ecorr = cost_func_ccsd(spec)
    penalty, cond = lindep_penalty_func(spec)
    spec.log_note('MCAO-cc-pVDZ basis: ehf= %.10f  eccsd= %.10f' % (ehf, ecorr))
    spec.log_note('                    penalty= %.3e  cond= %.3e  (overall)' % (penalty, cond))
    for ldp_func,(fml,fvasp) in zip(lindep_penalty_funcs,reference_solids):
        penalty, cond = ldp_func(spec)
        spec.log_note('                    penalty= %.3e  cond= %.3e  (%s)' % (penalty, cond, fml))
    spec.log_note('')

    spec.log_note('Initial cc-pVDZ basis:')
    spec_init.dump_basis()
    spec.log_note('')
    spec.log_note('MCAO-cc-pVDZ basis (with kappa0= %.3e):' % kappa0)
    spec.dump_basis()

    ''' Discussion of the optimized exponents:

        The initial DZ-quality basis is

            #BASIS SET: (3s,3p,1d) -> [3s,3p,1d]
            Al    S
                  0.9909710     1.0000000
            Al    S
                  0.1761703     1.0000000
            Al    S
                  0.0634298     1.0000000
            Al    P
                  0.2283400     1.0000000
            Al    P
                  0.0852216     1.0000000
            Al    P
                  0.0335128     1.0000000
            Al    D
                  0.1890000     1.0000000

        whereas the MCAO-optimized basis with `kappa0 = 1e8` is

            #BASIS SET: (3s,3p,1d) -> [3s,3p,1d]
            Al    S
                  0.9874518     1.0000000
            Al    S
                  0.1781855     1.0000000
            Al    S
                  0.0642899     1.0000000
            Al    P
                  4.5698848     1.0000000
            Al    P
                  0.2094373     1.0000000
            Al    P
                  0.0621026     1.0000000
            Al    D
                  0.1958799     1.0000000

        The s and d exponents change only modestly, whereas the p channel changes
        qualitatively: two p exponents remain in the valence range, while the third is
        driven to a much tighter value of approximately 4.57. The basis still formally
        contains three p primitives, but the tight primitive is likely to contribute
        little to the valence space described with the GTH-HF-rev pseudopotential. This
        behavior suggests that MCAO can expose redundancy in the initial basis; a
        subsequent basis-reduction calculation could test whether the tight primitive
        can be removed without sacrificing the target accuracy.
    '''
