''' This example demonstrates how to reproduce the primitive valence (s and p) exponents
    of the nitrogen cc-pVTZ basis from scratch.

    The first stage restricts each channel to an even-tempered form. This reduces the
    number of independent parameters and makes the initial optimization more stable.
    The optimized ETB is then converted to a fully parameterized basis and relaxed without
    the even-tempered constraint.

    NOTE: This example relies on PySCF for HF calculations.
'''

from pygto import lib
from pygto.basis import BasisSpec
from pygto.optimizer import ScheduledOptimizer
from pygto.data.elements import get_spin

from pyscf import gto, scf


if __name__ == '__main__':
    atm = 'N'
    spin = get_spin(atm)    # ground-state spin; should be 3
    val_l = [0,1]   # s, p

    ''' Construct the ROHF total-energy cost function.
    '''
    cost_func = lib.pyscf_helper.get_cost_func(
        atm, scf.ROHF, mol_settings={'spin':spin},
        keep_l=val_l,   # Not required here, but explicitly documents the channels
                        # included in the energy evaluation.
    )
    # The generated cost function is equivalent to the following implementation:
    # def cost_func(spec):
    #     basis = spec.get_pyscf_basis(keep_l=val_l)
    #     mol = gto.M(atom=atm, basis=basis, spin=spin).set(verbose=0)
    #     mf = scf.ROHF(mol)
    #     mf.kernel()
    #     return mf.e_tot

    cost = {}
    ''' Initialize an ETB with reasonable starting parameters.

        The nitrogen cc-pVTZ basis contains 10 s-type and 5 p-type primitives.
    '''
    etb_params = [
        # l, nprim, amin, beta
        (0, 10, 0.1, 3.5),
        (1, 5, 0.1, 3.5),
    ]
    spec_etb = BasisSpec.init_from_etb_params(etb_params, atm)
    cost['etb_init'] = cost_func(spec_etb)

    ''' Optimize the initial ETB while retaining the even-tempered constraint.
    '''
    opt_etb = ScheduledOptimizer(spec_etb, cost_func).set(verbose=4)
    opt_etb.kernel()
    cost['etb_opt'] = opt_etb.cost

    ''' Convert to independent exponents and perform a fully relaxed optimization.
    '''
    spec = spec_etb.convert_to('full')
    opt = ScheduledOptimizer(spec, cost_func).set(verbose=4)
    opt.kernel()
    cost['opt'] = opt.cost

    ''' Compare the optimized result with the reference cc-pVTZ basis.

        Reference output:

            **** Summary ***
            Init ETB  HF energy= -54.3823221735
            Opt  ETB  HF energy= -54.3932617147  ∆E= -0.0109395413
            Full opt  HF energy= -54.3973581243  ∆E= -0.0040964096
            Reference HF energy= -54.3973578474  ∆E=  0.0000002769

        Both the HF energy and the fully optimized Gaussian exponents closely reproduce
        the reference basis.
    '''
    spec_ref = BasisSpec.init_from_basis('cc-pvtz', atm, keep_l=val_l)
    cost['ref'] = cost_func(spec_ref)

    spec.log_note('')
    spec.log_note('**** Summary ***')
    spec.log_note('Init ETB  HF energy= %.10f' % (cost['etb_init']))
    spec.log_note('Opt  ETB  HF energy= %.10f  ∆E= % .10f' % (
        cost['etb_opt'], cost['etb_opt']-cost['etb_init']))
    spec.log_note('Full opt  HF energy= %.10f  ∆E= % .10f' % (
        cost['opt'], cost['opt']-cost['etb_opt']))
    spec.log_note('Reference HF energy= %.10f  ∆E= % .10f' % (
        cost['ref'], cost['ref']-cost['opt']))
    spec.log_note('')
    spec.log_note('Full optimized basis:')
    spec.dump_basis()
    spec.log_note('')
    spec.log_note('Reference basis:')
    spec_ref.dump_basis()
