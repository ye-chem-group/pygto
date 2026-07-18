''' This example demonstrates ScheduledOptimizer, the *recommended default optimizer*
    for basis-set optimization in PyGTO.

    ScheduledOptimizer applies a sequence of complementary optimization stages. Its
    default schedule first performs a derivative-free Nelder-Mead optimization and then
    refines the result with BFGS. This provides a more robust general-purpose workflow
    than selecting a single optimizer manually.

    For most calculations, we recommend using the default schedule and optimizer
    settings, as shown below. Customized stages and low-level optimizer settings are
    available for advanced use but should rarely be necessary.

    For the use of ScheduledOptimizer in practical basis set optimization tasks, see
    - "21_ccpvtz_val.py"
    - "22_ccpvtz_pol.py"
    - "23_ccpvtz_full.py"

    NOTE: This example relies on PySCF for RHF calculations.
'''

from pygto import lib
from pygto.basis import BasisSpec
from pygto.optimizer import ScheduledOptimizer

from pyscf import scf


if __name__ == '__main__':
    atm = 'He'
    spin = 0

    ''' Construct the RHF total-energy cost function.
    '''
    cost_func = lib.pyscf_helper.get_cost_func(
        atm, scf.RHF, mol_settings={'spin':spin}, keep_l=[0],
    )

    ''' Generate an initial three-primitive s basis with independent exponents.
    '''
    etb_params = [
        # l, nprim, amin, beta
        (0, 3, 0.3, 3.5),
    ]
    spec = BasisSpec.init_from_etb_params(etb_params, atm).convert_to('full')
    spec_init = spec.copy()
    cost_init = cost_func(spec)

    ''' Run the recommended default optimization.

        No stage or optimizer settings are required. The equivalent explicit
        constructor is `ScheduledOptimizer(spec, cost_func, stages='default')`.
        As with the individual optimizers, the input `spec` is updated in place.
    '''
    opt = ScheduledOptimizer(spec, cost_func)
    cost_opt, spec_opt = opt.kernel()

    spec.log_note('**** Summary ****')
    spec.log_note('Initial RHF energy=   %.10f' % cost_init)
    spec.log_note('Optimized RHF energy= %.10f' % cost_opt)
    spec.log_note('converged= %s  function evaluations= %d' % (
        opt.converged, opt.feval))
    for istage, result in enumerate(opt.history):
        spec.log_note('stage= %d  optimizer= %10s  cost= %.10f  converged= %s' % (
            istage+1, result['optimizer'].ljust(10), result['cost'], result['converged']))
    spec.log_note('')
    spec.log_note('Initial basis:')
    spec_init.dump_basis()
    spec.log_note('')
    spec.log_note('Optimized basis:')
    spec_opt.dump_basis()
