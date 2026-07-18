''' This example demonstrates how to use the TCAO workflow to reduce a basis while
    retaining a target accuracy defined by a supplied cost function.

    Specifically, we generate a compact valence basis for carbon with the ccECP. We start
    from the s and p primitives of the all-electron cc-pVQZ basis, then systematically
    remove unnecessary primitives and reoptimize the retained exponents. The allowed
    increase in the ROHF energy is `1e-4` Ha, or 0.1 mHa.

    NOTE: This example relies on PySCF for ROHF calculations.
'''

from pygto import lib
from pygto.basis import BasisSpec
from pygto.workflow import TCAO
from pygto.data.elements import get_spin

from pyscf import scf


if __name__ == '__main__':
    atm = 'C'
    spin = get_spin(atm)
    basis = 'cc-pvqz'
    val_l = [0,1]
    ecp = 'ccecp'
    ftol = 1e-4     # energy accuracy; unit is Hartree

    ''' Construct the ROHF total-energy cost function for carbon with the ccECP.
    '''
    cost_func = lib.pyscf_helper.get_cost_func(
        atm, scf.ROHF, mol_settings={'spin':spin, 'ecp':ecp}, keep_l=val_l,
    )

    cost = {}
    ''' Use the valence angular-momentum channels of all-electron cc-pVQZ as the
        deliberately oversized initial basis.
    '''
    spec = BasisSpec.init_from_basis(basis, atm,
        keep_l=val_l,   # Retain only the s and p channels.
    )
    cost['init'] = cost_func(spec)

    ''' Perform Target Cost Atomic Optimization (TCAO).
    '''
    spec_init = spec.copy() # save init basis
    opt = TCAO(spec, cost_func, ftol=ftol).set(verbose=5)
    opt.kernel()
    cost['opt'] = opt.cost

    ''' Compare the optimized valence basis with the s and p subsets of the
        ccECP-cc-pV(D/T/Q)Z families.

        Reference output:

            **** Accuracy and Size comparison ****
            Initial cc-pVQZ basis   HF energy= -5.3137977297  nao= 30  structure= 12s,6p
            Optimized TCAO basis    HF energy= -5.3142247149  nao= 23  structure= 5s,6p
            Reference ccECP-cc-pVDZ HF energy= -5.3142511552  nao= 40  structure= 10s,10p
            Reference ccECP-cc-pVTZ HF energy= -5.3142518267  nao= 44  structure= 11s,11p
            Reference ccECP-cc-pVQZ HF energy= -5.3142521634  nao= 48  structure= 12s,12p

        TCAO constrains the energy increase relative to its optimized initial basis,
        rather than relative to any external reference. Here, the resulting HF energy
        also lies within `ftol` of the ccECP-cc-pVXZ results while using far fewer
        primitives, showing that a compact valence set is sufficient for this atomic
        target.
    '''
    spec.log_note('**** Accuracy and Size comparison ****')
    spec.log_note('Inititial cc-pVQZ basis HF energy= %.10f  nao= %d  structure= %s' % (
        cost['init'], spec_init.nao, spec_init.structure
    ))
    spec.log_note('Optimized TCAO basis    HF energy= %.10f  nao= %d  structure= %s' % (
        cost['opt'], spec.nao, spec.structure
    ))
    for zeta in ['DZ','TZ','QZ']:
        spec_ref = BasisSpec.init_from_basis(f'ccecp-cc-pv{zeta}', atm, keep_l=val_l)
        spec.log_note('Reference ccECP-cc-pV%s HF energy= %.10f  nao= %d  structure= %s' % (
            zeta, cost_func(spec_ref), spec_ref.nao, spec_ref.structure
        ))
    spec.log_note('')
    spec.log_note('Optimized TCAO basis:')
    spec.dump_basis()
