''' This example demonstrates the basic use of a single named optimizer.

    We optimize a small even-tempered s basis for the helium atom using the
    derivative-free Nelder-Mead optimizer. The example also shows how to select a
    convergence-accuracy preset, limit the number of optimization cycles, control
    logging verbosity, and access the optimized BasisSpec.

    NOTE: This example relies on PySCF for RHF calculations.
'''

from pygto import lib
from pygto.basis import BasisSpec
from pygto.optimizer import NelderMead

from pyscf import scf


if __name__ == '__main__':
    atm = 'He'
    spin = 0

    ''' Construct the RHF total-energy cost function.
    '''
    cost_func = lib.pyscf_helper.get_cost_func(
        atm, scf.RHF, mol_settings={'spin':spin}, keep_l=[0],
    )

    ''' Initialize a three-primitive even-tempered s basis.

        The ETB representation has only two optimization parameters, `amin` and
        `beta`, making it a convenient small example for Nelder-Mead.
    '''
    etb_params = [
        # l, nprim, amin, beta
        (0, 3, 0.3, 3.5),
    ]
    spec = BasisSpec.init_from_etb_params(etb_params, atm)
    spec_init = spec.copy()
    cost_init = cost_func(spec)

    ''' Configure and run the Nelder-Mead optimizer.

        `accuracy` accepts "low", "medium", "high", or "ultra" and sets the
        convergence tolerances accordingly. With `verbose=4`, the optimizer prints
        its settings, basis data, and one line per optimization cycle. Use `verbose=3`
        for a shorter summary or `verbose=5` for debug-level output.

        NOTE: The explicit settings below are included for demonstration. For most
        calculations, the defaults are recommended; settings such as `max_cycle` and
        `max_inner` should generally be changed only by experienced users.
    '''
    opt = NelderMead(spec, cost_func).set(
        accuracy='medium',
        max_cycle=100,
        max_inner=50,
        verbose=4,
    )
    cost_opt, spec_opt = opt.kernel()

    ''' The optimizer mutates the original BasisSpec in place.

        `spec_opt`, `opt.spec`, and the original `spec` therefore refer to the same
        object. A copy must be made before optimization when the initial basis is
        needed later for comparison.
    '''
    assert spec_opt is spec
    assert opt.spec is spec

    spec.log_note('**** Summary ****')
    spec.log_note('Initial RHF energy=   %.10f' % cost_init)
    spec.log_note('Optimized RHF energy= %.10f' % cost_opt)
    spec.log_note('converged= %s  cycles= %d  function evaluations= %d' % (
        opt.converged, opt.cycle, opt.feval))
    spec.log_note('')
    spec.log_note('Initial basis:')
    spec_init.dump_basis()
    spec.log_note('')
    spec.log_note('Optimized basis:')
    spec.dump_basis()
