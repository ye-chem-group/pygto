''' This example demonstrates how to reproduce the polarization (d and f) exponents of
    the oxygen cc-pVTZ basis from scratch while keeping its valence basis fixed.

    NOTE: This example relies on PySCF for ROHF and CCSD calculations.
'''

from pygto import lib
from pygto.basis import BasisSpec
from pygto.optimizer import ScheduledOptimizer
from pygto.data.elements import get_spin

from pyscf import gto, scf, cc


if __name__ == '__main__':
    atm = 'O'
    spin = get_spin(atm)    # ground-state spin; should be 2
    val_l = [0,1]           # s, p
    pol_l = [2,3]           # d, f
    frozen = 1              # Freeze the 1s core orbital in the CCSD calculation.

    ''' Construct the CCSD correlation-energy cost function.
    '''
    cost_func = lib.pyscf_helper.get_cost_func(
        atm, scf.ROHF, mol_settings={'spin':spin},
        CORR=cc.CCSD, corr_settings={'frozen':frozen},
        # Do not set `keep_l=pol_l`: the CCSD calculation requires both the valence
        # and polarization channels even though only the latter will be optimized.
    )
    # The generated cost function is equivalent to the following implementation:
    # def cost_func(spec):
    #     basis = spec.get_pyscf_basis()
    #     mol = gto.M(atom=atm, basis=basis, spin=spin).set(verbose=0)
    #     mf = scf.ROHF(mol)
    #     mf.kernel()
    #     mc = cc.CCSD(mf, frozen=frozen)
    #     mc.kernel()
    #     return mc.e_corr

    cost = {}
    ''' Take the valence exponents from cc-pVTZ and append an initial ETB guess for
        the polarization functions.

        For a channel with at most two primitives, the ETB parameters can represent any
        ordered set of positive exponents. Because both polarization channels satisfy
        this condition, a separate fully parameterized optimization is unnecessary.
    '''
    etb_params = [
        # l, nprim, amin, beta
        (2, 2, 0.3, 3.5),
        (3, 1, 0.5, 3.5),
    ]
    spec_val = BasisSpec.init_from_basis('cc-pvtz', atm, keep_l=val_l)
    spec_pol = BasisSpec.init_from_etb_params(etb_params, atm)
    spec = spec_val.merge(spec_pol)
    cost['init'] = cost_func(spec)

    ''' Optimize only the initial polarization channels.

        The `temporary_active_l` context restricts optimization to the polarization
        channels. Without it, minimizing the CCSD correlation energy would also alter
        the valence exponents, which are intended to remain fixed in this example.
    '''
    with spec.temporary_active_l(pol_l):
        spec.log_warn('Inside `temporary_active_l`: active_l= %s' % (
            str(spec.active_l)))    # should see `pol_l`, i.e., [2,3]
        opt = ScheduledOptimizer(spec, cost_func).set(verbose=4)
        opt.kernel()
    spec.log_warn('Outside `temporary_active_l`: active_l= %s' % (
        str(spec.active_l)))        # should see None, which means all l-channels are active
    cost['opt'] = opt.cost

    ''' Compare the optimized polarization set with the reference cc-pVTZ basis.

        Reference output:

            **** Summary ***
            Init CCSD corr energy= -0.1505494069
            Opt  CCSD corr energy= -0.1690961466  ∆E= -0.0185467397
            Ref  CCSD corr energy= -0.1690663777  ∆E=  0.0000297689

        Both the CCSD correlation energy and the optimized polarization exponents are
        close to those obtained with the reference basis.
    '''
    spec_ref = BasisSpec.init_from_basis('cc-pvtz', atm)
    cost['ref'] = cost_func(spec_ref)

    spec.log_note('')
    spec.log_note('**** Summary ***')
    spec.log_note('Init CCSD corr energy= %.10f' % (cost['init']))
    spec.log_note('Opt  CCSD corr energy= %.10f  ∆E= % .10f' % (
        cost['opt'], cost['opt']-cost['init']))
    spec.log_note('Ref  CCSD corr energy= %.10f  ∆E= % .10f' % (
        cost['ref'], cost['ref']-cost['opt']))
    spec.log_note('')
    spec.log_note('Full optimized basis:')
    spec.dump_basis(keep_l=pol_l)       # selectively dump polarization shells
    spec.log_note('')
    spec.log_note('Reference basis:')
    spec_ref.dump_basis(keep_l=pol_l)   # selectively dump polarization shells
