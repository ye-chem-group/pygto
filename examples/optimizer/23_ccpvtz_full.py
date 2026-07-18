''' This example demonstrates how to reproduce the full carbon cc-pVTZ primitive set,
    including both valence and polarization exponents, from scratch.

    The valence exponents are optimized against the ROHF energy, whereas the polarization
    exponents are optimized against the CCSD correlation energy.

    NOTE: This example relies on PySCF for ROHF and CCSD calculations.
'''

from pygto import lib
from pygto.basis import BasisSpec
from pygto.optimizer import ScheduledOptimizer
from pygto.data.elements import get_spin

from pyscf import gto, scf, cc


if __name__ == '__main__':
    atm = 'C'
    spin = get_spin(atm)    # ground-state spin; should be 2
    val_l = [0,1]   # s, p
    pol_l = [2,3]   # d, f
    frozen = 1      # Freeze the 1s core orbital in the CCSD calculation.

    ''' Construct an ROHF total-energy cost function for the valence channels and a
        CCSD correlation-energy cost function for the polarization channels.
    '''
    cost_func_hf = lib.pyscf_helper.get_cost_func(
        atm, scf.ROHF, mol_settings={'spin':spin}, keep_l=val_l,
    )
    cost_func_ccsd = lib.pyscf_helper.get_cost_func(
        atm, scf.ROHF, mol_settings={'spin':spin},
        CORR=cc.CCSD, corr_settings={'frozen':frozen},
        # Do not set `keep_l=pol_l`: the CCSD calculation requires both the valence
        # and polarization channels.
    )

    ''' Generate reasonable starting exponents from ETB parameters and then convert
        immediately to a fully parameterized basis.

        The carbon cc-pVTZ basis contains 10 s-type, 5 p-type, 2 d-type, and 1 f-type
        primitives. The ETB construction supplies a well-spaced initial guess; conversion
        to `full` allows every exponent to vary independently during optimization.
    '''
    etb_param = [
        # l, nprim, amin, beta
        (0, 10, 0.1, 3.5),
        (1, 5, 0.1, 3.5),
        (2, 2, 0.3, 3.5),
        (3, 1, 0.5, 3.5),   # For nprim = 1, beta is ignored.
    ]
    spec = BasisSpec.init_from_etb_params(etb_param, atm).convert_to('full')

    ''' Optimize the valence channels against the ROHF energy.

        `temporary_active_l` ensures that only the valence subset is active.
    '''
    with spec.temporary_active_l(val_l):
        opt_hf = ScheduledOptimizer(spec, cost_func_hf).set(verbose=4)
        opt_hf.kernel()

    ''' Optimize the polarization channels against the CCSD correlation energy.

        `temporary_active_l` ensures that only the polarization subset is active.
    '''
    with spec.temporary_active_l(pol_l):
        opt_ccsd = ScheduledOptimizer(spec, cost_func_ccsd).set(verbose=4)
        opt_ccsd.kernel()

    ''' Compare the fully optimized basis with the reference cc-pVTZ basis.

        Reference output:

            **** Summary ***
            HF energy (optimized basis):       -37.6866624669
            HF energy (reference basis):       -37.6866622379  Difference:  0.0000002290
            CCSD corr energy (optimized basis): -0.0933666976
            CCSD corr energy (reference basis): -0.0933596761  Difference:  0.0000070215

        The HF energy, CCSD correlation energy, and optimized Gaussian exponents are all
        very close to their reference-basis values.
    '''
    spec_ref = BasisSpec.init_from_basis('cc-pvtz', atm)
    ehf_ref = cost_func_hf(spec_ref)
    eccsd_ref = cost_func_ccsd(spec_ref)

    spec.log_note('')
    spec.log_note('**** Summary ***')
    spec.log_note('HF energy (optimized basis):       %.10f' % (opt_hf.cost))
    spec.log_note('HF energy (reference basis):       %.10f  Difference: % .10f' % (
        ehf_ref, ehf_ref-opt_hf.cost))
    spec.log_note('CCSD corr energy (optimized basis): %.10f' % (opt_ccsd.cost))
    spec.log_note('CCSD corr energy (reference basis): %.10f  Difference: % .10f' % (
        eccsd_ref, eccsd_ref-opt_ccsd.cost))
    spec.log_note('')
    spec.log_note('Full optimized basis:')
    spec.dump_basis()
    spec.log_note('')
    spec.log_note('Reference basis:')
    spec_ref.dump_basis()
