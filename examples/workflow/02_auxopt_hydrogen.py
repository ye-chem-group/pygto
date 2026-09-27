''' This example demonstrates how to use the AuxOpt workflow to optimize an auxiliary
    basis for a specified orbital basis.

    Specifically, we optimize an auxiliary basis for hydrogen/cc-pVTZ and compare its
    errors and size with those of cc-pVTZ-JKFIT. An isolated hydrogen atom has only one
    electron and therefore has zero correlation energy, so its correlation metric cannot
    constrain auxiliary functions for the polarization channels. We therefore optimize
    the hydrogen auxiliary basis using H2 instead.

    NOTE: This example relies on PySCF for HF and MP2 calculations, for generating the
    initial AutoAux basis, and for providing the cc-pVTZ-JKFIT reference basis.
'''

from pygto import lib
from pygto.basis import BasisSpec
from pygto.workflow import AuxOpt

from pyscf import gto, scf, df


def format_cost_vec(cost_details):
    return ' '.join([
        f'{scaled_error:.3e}'
        for name, error, scaled_error in cost_details
    ])


if __name__ == '__main__':
    atm = 'H'
    atom = 'H 0 0 0; H 0 0 0.7414'
    spin = 0
    aobasis = 'cc-pvtz'

    ''' Construct the auxiliary-basis cost function for H2.

        cost = max(
            HF Coulomb matrix error * gamma_vjk,
            HF exchange matrix error * gamma_vjk,
            HF Coulomb energy error,
            HF exchange energy error,
            MP2 correlation energy error,
            same-spin T2 error * gamma_mp2,
            opposite-spin T2 error * gamma_mp2,
        )

        The matrix- and energy-error terms have different numerical scales. The
        matrix errors are therefore multiplied by `gamma_vjk`, whose default value
        is 0.1.
        The normalized T2-error terms are multiplied by `gamma_mp2`, whose default
        value is 10.
    '''
    cost_func = lib.pyscf_helper.get_cost_func_auxopt(
        atom,   # Use the H2 geometry rather than an isolated H atom.
        aobasis, scf.ROHF, mol_settings={'spin':spin}
    )

    cost = {}
    ''' Generate an initial atomic auxiliary basis with PySCF AutoAux.

        AutoAux supplies broad initial exponent ranges and angular-momentum coverage.
        AuxOpt converts each channel to a Legendre representation before optimization,
        which reduces the number of independent parameters for large channels while
        retaining a flexible exponent distribution.
    '''
    auxbasis = df.autoaux(gto.M(atom=atm, basis=aobasis, spin=None))[atm]
    spec_init = BasisSpec.init_from_basis(auxbasis, atm)
    cost['init'] = cost_func(spec_init, True)

    ''' Optimize and reduce the initial auxiliary basis to the default target error
        of `1e-5`.
    '''
    spec = spec_init.copy()
    opt = AuxOpt(spec, cost_func).set(verbose=5)
    opt.kernel()
    cost['opt'] = (opt.cost, opt.cost_details)

    ''' Compare with the reference cc-pVTZ-JKFIT auxiliary basis.

        Reference output:

**** H2 Accuracy ****
Init AutoAux cost= 2.838e-06  cost_vec= 4.004e-07 2.838e-06 9.757e-08 4.878e-08 2.579e-06 0.000e+00 2.135e-07
Opt Legendre cost= 9.922e-06  cost_vec= 9.920e-06 3.721e-06 4.848e-06 2.424e-06 9.922e-06 0.000e+00 2.308e-06
Ref    JKFIT cost= 1.252e-04  cost_vec= 1.323e-06 6.499e-05 1.092e-06 5.461e-07 5.486e-05 0.000e+00 1.252e-04

**** AuxBasis Size ****
Init AutoAux nauxao=  52  structure= 11s,4p,3d,2f
Opt Legendre nauxao=  27  structure= 4s,2p,2d,1f
Ref    JKFIT nauxao=  30  structure= 4s,3p,2d,1f

        The optimized ETB and reference cc-pVTZ-JKFIT bases have comparable sizes, while
        the optimized ETB gives a smaller fitting error for H2. The initial AutoAux basis
        gives a still smaller error but uses substantially more auxiliary functions.
    '''
    spec_ref = BasisSpec.init_from_basis(f'{aobasis}-jkfit', atm)
    cost['ref'] = cost_func(spec_ref, True)

    spec.log_note('**** H2 Accuracy ****')
    spec.log_note('Init AutoAux cost= %.3e  cost_vec= %s' % (
        cost['init'][0], format_cost_vec(cost['init'][1]))
    )
    spec.log_note('Opt Legendre cost= %.3e  cost_vec= %s' % (
        cost['opt'][0], format_cost_vec(cost['opt'][1]))
    )
    spec.log_note('Ref    JKFIT cost= %.3e  cost_vec= %s' % (
        cost['ref'][0], format_cost_vec(cost['ref'][1]))
    )
    spec.log_note('')

    spec.log_note('**** AuxBasis Size ****')
    spec.log_note('Init AutoAux nauxao= %3d  structure= %s' % (
        spec_init.nao, spec_init.structure
    ))
    spec.log_note('Opt Legendre nauxao= %3d  structure= %s' % (
        spec.nao, spec.structure
    ))
    spec.log_note('Ref    JKFIT nauxao= %3d  structure= %s' % (
        spec_ref.nao, spec_ref.structure
    ))
    spec.log_note('')
