''' This example demonstrates how to use the TCAO workflow to reduce a basis while
    retaining a target accuracy defined by a supplied cost function.

    Specifically, we generate a compact valence basis for aluminum with the gth-hf-rev
    pseudopotential. We start from the s and p primitives of the all-electron cc-pVQZ
    basis, then systematically remove unnecessary primitives and reoptimize the retained
    exponents. The allowed increase in the ROHF energy is `5e-4` Ha, or 0.5 mHa.

    NOTE: This example relies on PySCF for ROHF calculations.
'''

from pygto import lib
from pygto.basis import BasisSpec
from pygto.workflow import TCAO
from pygto.data.elements import get_spin

from pyscf import scf


if __name__ == '__main__':
    atm = 'Al'
    spin = get_spin(atm)
    basis = 'cc-pvqz'
    val_l = [0,1]
    pseudo = 'gth-hf-rev'
    ftol = 5e-4     # Target energy-increase tolerance in Hartree.

    ''' Construct the ROHF total-energy cost function for aluminum with the
        GTH-HF-rev pseudopotential.
    '''
    cost_func = lib.pyscf_helper.get_cost_func(
        atm, scf.ROHF, mol_settings={'spin':spin, 'pseudo':pseudo}, keep_l=val_l,
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
    spec_init = spec.copy() # Save the initial basis for comparison.
    opt = TCAO(spec, cost_func, ftol=ftol).set(verbose=5)
    opt.kernel()
    cost['opt'] = opt.cost

    ''' Compare the optimized valence basis with the s and p subsets of standard
        GTH basis sets.

        Reference output:

            **** Accuracy and Size comparison ****
            Initial cc-pVQZ basis HF energy= -1.8839990332  nao= 49  structure= 16s,11p
            Optimized TCAO basis  HF energy= -1.8836488740  nao= 12  structure= 3s,3p
            Reference GTH-DZVP    HF energy= -1.8836514015  nao= 16  structure= 4s,4p
            Reference GTH-TZV2P   HF energy= -1.8837204538  nao= 20  structure= 5s,5p
    '''
    spec.log_note('**** Accuracy and Size comparison ****')
    spec.log_note('Initial cc-pVQZ basis HF energy= %.10f  nao= %d  structure= %s' % (
        cost['init'], spec_init.nao, spec_init.structure
    ))
    spec.log_note('Optimized TCAO basis  HF energy= %.10f  nao= %d  structure= %s' % (
        cost['opt'], spec.nao, spec.structure
    ))
    for name in ['GTH-SZV', 'GTH-DZVP', 'GTH-TZV2P']:
        spec_ref = BasisSpec.init_from_basis(name, atm, keep_l=val_l)
        spec.log_note('Reference %11s HF energy= %.10f  nao= %d  structure= %s' % (
            name.ljust(11), cost_func(spec_ref), spec_ref.nao, spec_ref.structure
        ))
    spec.log_note('')
    spec.log_note('Optimized TCAO basis:')
    spec.dump_basis()
