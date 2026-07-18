''' This example demonstrates direct basis optimization with the BFGS optimizer.

    We optimize three independent s exponents for the helium atom. In addition to the
    common accuracy and verbosity settings, the example shows a BFGS-specific step-size
    setting and how to inspect convergence information after optimization.

    NOTE: This example relies on PySCF for RHF calculations.
'''

import numpy as np

from pygto import lib
from pygto.basis import BasisSpec
from pygto.optimizer import BFGS

from pyscf import scf


if __name__ == '__main__':
    atm = 'He'
    spin = 0

    ''' Construct the RHF total-energy cost function.
    '''
    cost_func = lib.pyscf_helper.get_cost_func(
        atm, scf.RHF, mol_settings={'spin':spin}, keep_l=[0],
    )

    ''' Generate an initial ETB and convert it to independent exponents.

        Conversion to `full` removes the even-tempered constraint, so BFGS optimizes
        all three exponents independently.
    '''
    etb_params = [
        # l, nprim, amin, beta
        (0, 3, 0.3, 3.5),
    ]
    spec = BasisSpec.init_from_etb_params(etb_params, atm).convert_to('full')
    spec_init = spec.copy()
    cost_init = cost_func(spec)

    ''' Configure and run the BFGS optimizer.

        Because no `grad_func` is supplied, BFGS evaluates gradients numerically.
        The named `accuracy` preset controls `ftol`, `xtol`, and `gtol`, while
        `max_step` limits the largest component of a proposed BFGS search direction.
        Here, `verbose=3` prints only the principal results; use `verbose=4` to see
        optimizer settings and cycle-by-cycle progress.

        NOTE: The explicit settings below are included for demonstration. For most
        calculations, the defaults are recommended; settings such as `max_cycle` and
        `max_step` should generally be changed only by experienced users.
    '''
    opt = BFGS(spec, cost_func).set(
        accuracy='medium',
        max_cycle=100,
        max_step=0.25,
        verbose=3,
    )
    opt.kernel()

    ''' Access the optimized basis and convergence information through the optimizer.
    '''
    spec_opt = opt.spec
    gmax = np.max(np.abs(opt.gradient))

    spec.log_note('**** Summary ****')
    spec.log_note('Initial RHF energy=   %.10f' % cost_init)
    spec.log_note('Optimized RHF energy= %.10f' % opt.cost)
    spec.log_note('converged= %s  stop_reason= %s' % (
        opt.converged, opt.stop_reason))
    spec.log_note('cycles= %d  function evaluations= %d  max gradient= %.3e' % (
        opt.cycle, opt.feval, gmax))
    spec.log_note('')
    spec.log_note('Initial basis:')
    spec_init.dump_basis()
    spec.log_note('')
    spec.log_note('Optimized basis:')
    spec_opt.dump_basis()
